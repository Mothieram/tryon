"""
hybrid_warper.py - Hybrid Cloth Warping Engine
Core warping pipeline using Thin Plate Spline (TPS) deformation,
shoulder alignment, sleeve deformation, and physics-inspired motion lag.
This is the heart of the realistic shirt overlay system.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List, Dict
from scipy.interpolate import RBFInterpolator
from dataclasses import dataclass

from engine.utils import (
    setup_logger,
    PoseKeypoints,
    smooth_array,
    smooth_value,
    point_distance,
    rotate_point,
)
from engine.garment_landmarks import GarmentLandmarks

logger = setup_logger("hybrid_warper")


@dataclass
class WarpResult:
    """Output from the warp engine."""
    warped_shirt: np.ndarray       # BGRA warped shirt image
    placement_x: int               # X offset in frame
    placement_y: int               # Y offset in frame
    scale: float                   # Applied scale factor
    rotation: float                # Applied rotation in degrees
    target_width: int
    target_height: int
    confidence: float = 1.0


class HybridWarper:
    """
    Hybrid Cloth Warping Engine.

    Combines multiple warping strategies:
    1. Rigid transformation (scale + rotate) for global fit
    2. TPS local deformation for realistic cloth follow
    3. Sleeve deformation following elbow/wrist positions
    4. Physics-inspired motion lag for natural cloth movement
    5. Frame-to-frame smoothing for stable rendering

    Pipeline per frame:
    ─────────────────────────────────────────────────
    Pose Keypoints → Compute target control points
                   → Match against garment landmarks
                   → Compute TPS warp field
                   → Apply rigid + warp to shirt image
                   → Apply sleeve deformation
                   → Smooth with previous frame
    ─────────────────────────────────────────────────
    """

    def __init__(
        self,
        smooth_alpha: float = 0.35,
        physics_lag: float = 0.25,
        max_scale: float = 2.5,
        min_scale: float = 0.3,
        tps_smooth: float = 0.5,
    ):
        """
        Args:
            smooth_alpha: Temporal smoothing of warp parameters
            physics_lag: Cloth momentum (higher = more lag)
            max_scale: Maximum allowed scale
            min_scale: Minimum allowed scale
            tps_smooth: TPS regularization (0=exact interpolation, higher=smoother)
        """
        self.smooth_alpha = smooth_alpha
        self.physics_lag = physics_lag
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.tps_smooth = tps_smooth

        # Smoothed state
        self._prev_scale: float = 1.0
        self._prev_rotation: float = 0.0
        self._prev_offset: np.ndarray = np.array([0.0, 0.0])
        self._prev_ctrl_pts: Optional[np.ndarray] = None

        # Physics state (velocity for motion lag)
        self._velocity_offset: np.ndarray = np.zeros(2)
        self._velocity_scale: float = 0.0

        # Cache last warp for static poses
        self._last_pose_hash: float = 0.0
        self._last_warp: Optional[WarpResult] = None

    def warp(
        self,
        shirt_image: np.ndarray,
        landmarks: GarmentLandmarks,
        pose: PoseKeypoints,
        frame_shape: Tuple[int, int],
    ) -> Optional[WarpResult]:
        """
        Warp shirt image to fit detected body pose.

        Args:
            shirt_image: BGRA shirt PNG
            landmarks: Garment control points
            pose: Detected body pose
            frame_shape: (height, width) of camera frame

        Returns:
            WarpResult or None if pose insufficient
        """
        if not pose or not pose.is_usable(min_keypoints=4):
            return None

        fh, fw = frame_shape[:2]

        # ─── Step 1: Compute rigid transform parameters ──────────
        scale, rotation, offset = self._compute_rigid_transform(
            landmarks, pose, fw, fh
        )

        # ─── Step 2: Apply temporal smoothing ────────────────────
        scale = self._smooth_scale(scale)
        rotation = self._smooth_rotation(rotation)
        offset = self._smooth_offset(offset)

        # ─── Step 3: Resize & rotate shirt ───────────────────────
        sh, sw = shirt_image.shape[:2]
        target_w = int(sw * scale)
        target_h = int(sh * scale)

        if target_w < 10 or target_h < 10:
            return None

        # Resize shirt
        resized = cv2.resize(
            shirt_image,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Rotate shirt
        if abs(rotation) > 0.5:
            resized = self._rotate_image(resized, rotation)
            target_h, target_w = resized.shape[:2]

        # ─── Step 4: TPS local deformation ───────────────────────
        warped = self._apply_tps_warp(resized, landmarks, pose, scale, rotation)

        # ─── Step 5: Sleeve deformation ──────────────────────────
        warped = self._deform_sleeves(warped, landmarks, pose, scale)

        # ─── Step 6: Compute placement ───────────────────────────
        place_x = int(offset[0])
        place_y = int(offset[1])

        return WarpResult(
            warped_shirt=warped,
            placement_x=place_x,
            placement_y=place_y,
            scale=scale,
            rotation=rotation,
            target_width=warped.shape[1],
            target_height=warped.shape[0],
        )

    def _compute_rigid_transform(
        self,
        landmarks: GarmentLandmarks,
        pose: PoseKeypoints,
        fw: int,
        fh: int,
    ) -> Tuple[float, float, np.ndarray]:
        """Compute scale, rotation, and translation for global shirt fit."""
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip

        if not ls or not rs or not ls.valid or not rs.valid:
            return 1.0, 0.0, np.array([fw // 4, fh // 4], dtype=float)

        # ── Scale from shoulder width ─────────────────────────────
        body_shoulder_width = pose.shoulder_width
        shirt_shoulder_width = float(
            landmarks.shoulder_right[0] - landmarks.shoulder_left[0]
        )

        if shirt_shoulder_width < 1:
            shirt_shoulder_width = landmarks.width * 0.7

        scale = body_shoulder_width / shirt_shoulder_width if shirt_shoulder_width > 0 else 1.0

        # Scale correction: account for shirt having wider image than body part
        garment_w = landmarks.shirt_width
        scale_by_content = body_shoulder_width / max(garment_w, 1) * 1.1
        scale = max(scale, scale_by_content * 0.9)

        # Also consider torso height
        if lh and rh and lh.valid and rh.valid:
            body_torso_h = pose.torso_height
            shirt_torso_h = float(
                landmarks.hem_center[1] - landmarks.collar_center[1]
            )
            if shirt_torso_h > 0 and body_torso_h > 0:
                scale_h = body_torso_h / shirt_torso_h
                scale = (scale + scale_h) / 2  # Average of width and height scales

        scale = np.clip(scale, self.min_scale, self.max_scale)

        # ── Rotation from shoulder tilt ───────────────────────────
        rotation = pose.torso_angle  # degrees

        # ── Translation: align shirt shoulder to body shoulder ────
        body_shoulder_mid = np.array([
            (ls.x + rs.x) / 2,
            (ls.y + rs.y) / 2,
        ])

        # Scaled shirt collar should sit near body shoulder midpoint
        shirt_collar = np.array(landmarks.collar_center, dtype=float)
        shirt_collar_scaled = shirt_collar * scale

        # Offset = body_shoulder_mid - scaled_shirt_collar_y + upward shift
        upward_shift = body_shoulder_width * 0.05  # slight upward to cover shoulder seam
        offset = body_shoulder_mid - shirt_collar_scaled + np.array([0, -upward_shift])

        return float(scale), float(rotation), offset

    def _smooth_scale(self, new_scale: float) -> float:
        """EMA + physics lag for scale."""
        smoothed = smooth_value(self._prev_scale, new_scale, alpha=self.smooth_alpha)
        self._velocity_scale = smooth_value(
            self._velocity_scale, new_scale - self._prev_scale, alpha=0.4
        )
        result = smoothed + self._velocity_scale * self.physics_lag
        result = np.clip(result, self.min_scale, self.max_scale)
        self._prev_scale = result
        return float(result)

    def _smooth_rotation(self, new_rot: float) -> float:
        # Handle angle wrap-around
        diff = new_rot - self._prev_rotation
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        smoothed = self._prev_rotation + diff * self.smooth_alpha
        self._prev_rotation = smoothed
        return float(smoothed)

    def _smooth_offset(self, new_offset: np.ndarray) -> np.ndarray:
        """EMA + physics for position."""
        velocity = new_offset - self._prev_offset
        self._velocity_offset = smooth_array(
            self._velocity_offset, velocity, alpha=0.35
        )
        smoothed = smooth_array(self._prev_offset, new_offset, alpha=self.smooth_alpha)
        result = smoothed + self._velocity_offset * self.physics_lag
        self._prev_offset = result
        return result

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around center with transparent padding."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # Expand canvas to prevent cropping
        diagonal = int(np.sqrt(h**2 + w**2))
        pad_h = (diagonal - h) // 2
        pad_w = (diagonal - w) // 2

        # Pad with transparency
        padded = cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
        )

        ph, pw = padded.shape[:2]
        M = cv2.getRotationMatrix2D((pw // 2, ph // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            padded, M, (pw, ph),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0, 0, 0, 0],
        )

        # Crop back to original + small buffer
        buf = 5
        y1 = max(0, pad_h - buf)
        y2 = min(ph, pad_h + h + buf)
        x1 = max(0, pad_w - buf)
        x2 = min(pw, pad_w + w + buf)
        return rotated[y1:y2, x1:x2]

    def _apply_tps_warp(
        self,
        shirt: np.ndarray,
        landmarks: GarmentLandmarks,
        pose: PoseKeypoints,
        scale: float,
        rotation: float,
    ) -> np.ndarray:
        """
        Apply Thin Plate Spline local deformation.
        Corrects for body shape differences not captured by rigid transform.
        """
        sh, sw = shirt.shape[:2]

        # Build source control points (on resized shirt)
        src_pts = self._build_shirt_control_points(landmarks, scale, sw, sh)
        dst_pts = self._build_body_control_points(pose, landmarks, scale, rotation)

        if src_pts is None or dst_pts is None:
            return shirt

        # We need enough valid corresponding pairs
        if len(src_pts) < 4:
            return shirt

        try:
            # Compute displacement at each source point
            displacements = dst_pts - src_pts

            # Use RBF interpolation (TPS equivalent)
            rbf_x = RBFInterpolator(
                src_pts, displacements[:, 0],
                kernel="thin_plate_spline",
                smoothing=self.tps_smooth,
            )
            rbf_y = RBFInterpolator(
                src_pts, displacements[:, 1],
                kernel="thin_plate_spline",
                smoothing=self.tps_smooth,
            )

            # Build dense warp map
            step = 4  # Subsample for performance
            ys = np.arange(0, sh, step)
            xs = np.arange(0, sw, step)
            grid_x, grid_y = np.meshgrid(xs, ys)
            query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

            dx = rbf_x(query_pts).reshape(len(ys), len(xs))
            dy = rbf_y(query_pts).reshape(len(ys), len(xs))

            # Upsample displacement maps
            map_x_small = grid_x.astype(np.float32) + dx.astype(np.float32)
            map_y_small = grid_y.astype(np.float32) + dy.astype(np.float32)

            map_x = cv2.resize(map_x_small, (sw, sh), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y_small, (sw, sh), interpolation=cv2.INTER_LINEAR)

            # Apply remap
            warped = cv2.remap(
                shirt, map_x, map_y,
                cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[0, 0, 0, 0],
            )
            return warped

        except Exception as e:
            logger.debug(f"TPS warp failed: {e}")
            return shirt

    def _build_shirt_control_points(
        self,
        landmarks: GarmentLandmarks,
        scale: float,
        sw: int,
        sh: int,
    ) -> Optional[np.ndarray]:
        """Control points in scaled shirt image coordinates."""
        lm = landmarks
        pts = [
            lm.collar_center,
            lm.collar_left,
            lm.collar_right,
            lm.shoulder_left,
            lm.shoulder_right,
            lm.chest_center,
            lm.waist_center,
            lm.hem_center,
        ]

        # Scale from original shirt to current resized shirt
        orig_scale_x = sw / max(lm.width, 1)
        orig_scale_y = sh / max(lm.height, 1)

        result = []
        for p in pts:
            scaled_x = p[0] * orig_scale_x
            scaled_y = p[1] * orig_scale_y
            if 0 <= scaled_x < sw and 0 <= scaled_y < sh:
                result.append([scaled_x, scaled_y])

        return np.array(result, dtype=np.float32) if len(result) >= 4 else None

    def _build_body_control_points(
        self,
        pose: PoseKeypoints,
        landmarks: GarmentLandmarks,
        scale: float,
        rotation: float,
    ) -> Optional[np.ndarray]:
        """
        Target positions for shirt control points in shirt image space.
        Maps anatomical positions to where they should sit on the body.
        """
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip

        if not ls or not rs or not ls.valid or not rs.valid:
            return None

        sw_body = pose.shoulder_width
        # In shirt image space, we don't move shoulder pts much (rigid handles it)
        # But we adjust chest/waist/hem based on body proportions

        # Compute torso scaling factors
        shirt_torso_h = landmarks.hem_center[1] - landmarks.collar_center[1]
        body_torso_h = pose.torso_height if pose.torso_height > 5 else shirt_torso_h * scale

        scaled_shirt_h = shirt_torso_h * scale if shirt_torso_h > 0 else 200

        # Scale shoulder width difference
        shirt_sw = (landmarks.shoulder_right[0] - landmarks.shoulder_left[0])
        body_sw_in_shirt = sw_body / scale if scale > 0 else shirt_sw
        sw_factor = body_sw_in_shirt / max(shirt_sw, 1)

        lm = landmarks
        orig_scale_x = (lm.width * scale) / max(lm.width, 1)
        orig_scale_y = (lm.height * scale) / max(lm.height, 1)

        def scaled_pt(p):
            return np.array([p[0] * orig_scale_x, p[1] * orig_scale_y])

        # Collar stays at same position (rigid transform handles it)
        collar_c = scaled_pt(lm.collar_center)
        collar_l = scaled_pt(lm.collar_left)
        collar_r = scaled_pt(lm.collar_right)

        # Shoulders: widen/narrow based on body
        sh_l = scaled_pt(lm.shoulder_left)
        sh_r = scaled_pt(lm.shoulder_right)
        sh_mid = (sh_l + sh_r) / 2
        sh_l_adj = sh_mid + (sh_l - sh_mid) * sw_factor
        sh_r_adj = sh_mid + (sh_r - sh_mid) * sw_factor

        # Chest/waist/hem: adjust vertical stretch
        th_factor = body_torso_h / max(scaled_shirt_h, 1)
        th_factor = np.clip(th_factor, 0.6, 1.4)

        chest = scaled_pt(lm.chest_center)
        waist = scaled_pt(lm.waist_center)
        hem = scaled_pt(lm.hem_center)

        collar_y = collar_c[1]

        def stretch_y(pt):
            dy = pt[1] - collar_y
            return np.array([pt[0], collar_y + dy * th_factor])

        chest_adj = stretch_y(chest)
        waist_adj = stretch_y(waist)
        hem_adj = stretch_y(hem)

        pts = np.array([
            collar_c, collar_l, collar_r,
            sh_l_adj, sh_r_adj,
            chest_adj, waist_adj, hem_adj,
        ], dtype=np.float32)

        return pts

    def _deform_sleeves(
        self,
        shirt: np.ndarray,
        landmarks: GarmentLandmarks,
        pose: PoseKeypoints,
        scale: float,
    ) -> np.ndarray:
        """
        Deform sleeve areas to follow arm positions.
        Uses affine warp on sleeve ROIs.
        """
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        le = pose.left_elbow
        re = pose.right_elbow

        if not any([ls and ls.valid, rs and rs.valid]):
            return shirt

        result = shirt.copy()
        sh, sw = shirt.shape[:2]

        # Scale factors from landmark space to current shirt size
        sx = sw / max(landmarks.width, 1)
        sy = sh / max(landmarks.height, 1)

        # ── Left sleeve deformation ────────────────────────────────
        if ls and ls.valid and le and le.valid:
            result = self._warp_sleeve_region(
                result, landmarks,
                side="left",
                shoulder_kp=ls,
                elbow_kp=le,
                sx=sx, sy=sy,
                scale=scale,
            )

        # ── Right sleeve deformation ───────────────────────────────
        if rs and rs.valid and re and re.valid:
            result = self._warp_sleeve_region(
                result, landmarks,
                side="right",
                shoulder_kp=rs,
                elbow_kp=re,
                sx=sx, sy=sy,
                scale=scale,
            )

        return result

    def _warp_sleeve_region(
        self,
        shirt: np.ndarray,
        landmarks: GarmentLandmarks,
        side: str,
        shoulder_kp,
        elbow_kp,
        sx: float,
        sy: float,
        scale: float,
    ) -> np.ndarray:
        """Apply affine transform to one sleeve."""
        sh, sw = shirt.shape[:2]

        if side == "left":
            garment_shoulder = landmarks.shoulder_left
            garment_sleeve_end = landmarks.sleeve_left_end
        else:
            garment_shoulder = landmarks.shoulder_right
            garment_sleeve_end = landmarks.sleeve_right_end

        # Source points in shirt space
        src_shoulder = np.array([garment_shoulder[0] * sx, garment_shoulder[1] * sy], dtype=np.float32)
        src_sleeve = np.array([garment_sleeve_end[0] * sx, garment_sleeve_end[1] * sy], dtype=np.float32)
        src_mid = (src_shoulder + src_sleeve) / 2

        # Target: sleeve should follow elbow direction
        sh_pos = shoulder_kp.to_array()
        el_pos = elbow_kp.to_array()

        # Direction of arm in camera space
        arm_vec = el_pos - sh_pos
        arm_len = np.linalg.norm(arm_vec)
        if arm_len < 5:
            return shirt

        arm_dir = arm_vec / arm_len

        # Sleeve length in shirt
        sleeve_len = np.linalg.norm(src_sleeve - src_shoulder)

        # Target sleeve end = shirt shoulder + arm_direction * sleeve_length (in shirt space)
        # Convert body positions to shirt space
        target_sleeve = src_shoulder + arm_dir * sleeve_len

        # Slight bend at elbow
        target_mid = (src_shoulder + target_sleeve) / 2

        # Define triangle for affine warp
        src_tri = np.array([src_shoulder, src_sleeve, src_mid], dtype=np.float32)
        dst_tri = np.array([src_shoulder, target_sleeve, target_mid], dtype=np.float32)

        # Get affine matrix
        M = cv2.getAffineTransform(src_tri, dst_tri)

        # Apply only to sleeve region (bounding box)
        x_min = int(min(src_shoulder[0], src_sleeve[0], target_sleeve[0])) - 15
        x_max = int(max(src_shoulder[0], src_sleeve[0], target_sleeve[0])) + 15
        y_min = int(min(src_shoulder[1], src_sleeve[1], target_sleeve[1])) - 15
        y_max = int(max(src_shoulder[1], src_sleeve[1], target_sleeve[1])) + 15

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(sw, x_max)
        y_max = min(sh, y_max)

        if x_max <= x_min or y_max <= y_min:
            return shirt

        roi = shirt[y_min:y_max, x_min:x_max].copy()

        # Translate affine to ROI space
        M_roi = M.copy()
        M_roi[0, 2] -= x_min
        M_roi[1, 2] -= y_min

        roi_h, roi_w = roi.shape[:2]
        warped_roi = cv2.warpAffine(
            roi, M_roi, (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        # Blend back with feathered edge
        result = shirt.copy()
        # Create blend mask
        blend_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        # Fill sleeve area in blend mask
        sleeve_pts_roi = (np.array([
            src_shoulder, src_sleeve, target_sleeve, target_mid
        ]) - np.array([x_min, y_min])).astype(np.int32)
        cv2.fillConvexPoly(blend_mask, sleeve_pts_roi, 1.0)
        blend_mask = cv2.GaussianBlur(blend_mask, (11, 11), 0)

        blend_3 = np.stack([blend_mask] * 4, axis=2)
        original_roi = shirt[y_min:y_max, x_min:x_max].astype(np.float32)
        blended = original_roi * (1 - blend_3) + warped_roi.astype(np.float32) * blend_3
        result[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def reset(self):
        """Reset all smoothing state."""
        self._prev_scale = 1.0
        self._prev_rotation = 0.0
        self._prev_offset = np.array([0.0, 0.0])
        self._prev_ctrl_pts = None
        self._velocity_offset = np.zeros(2)
        self._velocity_scale = 0.0
        self._last_warp = None


__all__ = ["HybridWarper", "WarpResult"]
