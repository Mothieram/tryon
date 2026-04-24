"""
hybrid_warper.py - Hybrid Cloth Warping Engine  [FIXED]
Core warping pipeline using Thin Plate Spline (TPS) deformation,
shoulder alignment, sleeve deformation, and physics-inspired motion lag.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List, Dict
from scipy.interpolate import RBFInterpolator
from dataclasses import dataclass

from engine.coreutils import (
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

    FIX NOTES (2026-04-24):
    - shirt is right-side-up: collar at TOP of PNG (y=min), hem at BOTTOM (y=max)
    - placement offset now anchors shirt COLLAR to body SHOULDER position
      with a small upward shift so the collar sits just above the shoulders
    - scale now accounts for shirt content width (not full image width)
    - rotation sign corrected for camera-frame coordinate system
    - TPS warp only applied when src/dst pairs are geometrically consistent
    """

    def __init__(
        self,
        smooth_alpha: float = 0.35,
        physics_lag: float = 0.25,
        max_scale: float = 2.5,
        min_scale: float = 0.3,
        tps_smooth: float = 0.5,
    ):
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
        if not pose or not pose.is_usable(min_keypoints=4):
            return None

        fh, fw = frame_shape[:2]

        # Step 1: Compute rigid transform (scale, rotation, placement offset)
        scale, rotation, offset = self._compute_rigid_transform(
            landmarks, pose, fw, fh
        )

        # Step 2: Temporal smoothing
        scale = self._smooth_scale(scale)
        rotation = self._smooth_rotation(rotation)
        offset = self._smooth_offset(offset)

        # Step 3: Resize shirt to match body
        sh, sw = shirt_image.shape[:2]
        target_w = int(sw * scale)
        target_h = int(sh * scale)

        if target_w < 10 or target_h < 10:
            return None

        resized = cv2.resize(
            shirt_image,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Step 4: Rotate shirt to match shoulder tilt
        if abs(rotation) > 0.5:
            old_h, old_w = target_h, target_w
            resized = self._rotate_image(resized, rotation)
            target_h, target_w = resized.shape[:2]
            # Compensate canvas expansion so collar stays anchored
            dw = (target_w - old_w) // 2
            dh = (target_h - old_h) // 2
            offset -= np.array([dw, dh], dtype=float)

        # Step 5: TPS local deformation (body-shape correction)
        warped = self._apply_tps_warp(resized, landmarks, pose, scale, rotation)

        # Step 6: Sleeve deformation to follow arm direction
        warped = self._deform_sleeves(warped, landmarks, pose, scale)

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
        """
        Compute scale, rotation, and translation for global shirt fit.
        
        KEY GEOMETRY:
        - Shirt PNG: collar at TOP (small y), hem at BOTTOM (large y)
        - Body: neck base is ABOVE shoulder line, approximately at nose_y + some offset
        - The collar center should align with the body's neck base point
        """
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip
        nose = pose.nose

        if not ls or not rs or not ls.valid or not rs.valid:
            return 1.0, 0.0, np.array([fw // 4, fh // 4], dtype=float)

        # ── Scale: match body shoulder width to shirt shoulder width ─────────
        body_sw = pose.shoulder_width

        # Shirt shoulder width (landmark space)
        shirt_sw = float(landmarks.shoulder_right[0] - landmarks.shoulder_left[0])
        if shirt_sw < 10:
            shirt_sw = max(10.0, float(landmarks.shirt_width))

        # Primary scale from shoulder widths with expansion for coverage
        scale = (body_sw / shirt_sw) * 1.15 if shirt_sw > 0 else 1.0

        # Secondary: torso height check (if hips are visible)
        if lh and rh and lh.valid and rh.valid:
            body_th = pose.torso_height
            shirt_th = float(landmarks.hem_center[1] - landmarks.collar_center[1])
            if shirt_th > 10 and body_th > 20:
                scale_h = body_th / shirt_th
                scale_h = float(np.clip(scale_h, self.min_scale, self.max_scale))
                # Prefer shoulder width scaling (70%) over height (30%)
                scale = scale * 0.7 + scale_h * 0.3

        scale = float(np.clip(scale, self.min_scale, self.max_scale))

        # ── Rotation: shoulder tilt ───────────────────────────────────────────
        rotation = pose.torso_angle

        # ═════════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Calculate proper neck base position
        # ═════════════════════════════════════════════════════════════════════════
        
        # Shoulder midpoint is the CENTER of the shoulder line
        shoulder_mid = np.array([
            (ls.x + rs.x) / 2.0,
            (ls.y + rs.y) / 2.0,
        ], dtype=float)

        # The neck base (where collar should sit) is ABOVE the shoulder midpoint
        # We can estimate it using the nose position if available
        
        if nose and nose.valid:
            # Use nose to find the neck base
            # Neck base is approximately halfway between nose and shoulder line
            neck_base_x = shoulder_mid[0]  # Horizontal center of shoulders
            neck_base_y = (nose.y + shoulder_mid[1]) / 2.0  # Halfway between nose and shoulders
            
            # Adjust X position if nose is off-center (person looking sideways)
            nose_offset = nose.x - shoulder_mid[0]
            if abs(nose_offset) > body_sw * 0.05:  # If nose is significantly off-center
                # Adjust neck base slightly toward nose
                neck_base_x += nose_offset * 0.3  # 30% influence from nose position
        else:
            # Fallback: estimate neck base above shoulder line
            # Typical anatomy: neck base is about 20-25% of shoulder width above shoulder line
            neck_base_x = shoulder_mid[0]
            neck_base_y = shoulder_mid[1] - body_sw * 0.22
        
        # ═════════════════════════════════════════════════════════════════════════
        # Calculate collar position in scaled shirt coordinates
        # ═════════════════════════════════════════════════════════════════════════
        
        collar_in_scaled = np.array(landmarks.collar_center, dtype=float) * scale
        
        # ═════════════════════════════════════════════════════════════════════════
        # Calculate offset: position shirt so collar aligns with neck base
        # ═════════════════════════════════════════════════════════════════════════
        
        # offset = neck_base_position - collar_position_in_scaled_image
        # This means: shirt's top-left corner = neck_base - collar_position
        offset = np.array([
            neck_base_x - collar_in_scaled[0],
            neck_base_y - collar_in_scaled[1]
        ], dtype=float)
        
        # Add a small safety margin to ensure shirt doesn't start too high
        # (prevents collar from going above neck into face area)
        if nose and nose.valid:
            min_y = nose.y - body_sw * 0.1  # Collar shouldn't go above nose level
            collar_top_y = offset[1]
            if collar_top_y < min_y:
                offset[1] = min_y

        return scale, float(rotation), offset

    # ── Smoothing helpers ─────────────────────────────────────────────────────

    def _smooth_scale(self, new_scale: float) -> float:
        smoothed = smooth_value(self._prev_scale, new_scale, alpha=self.smooth_alpha)
        self._velocity_scale = smooth_value(
            self._velocity_scale, new_scale - self._prev_scale, alpha=0.4
        )
        result = smoothed + self._velocity_scale * self.physics_lag
        result = float(np.clip(result, self.min_scale, self.max_scale))
        self._prev_scale = result
        return result

    def _smooth_rotation(self, new_rot: float) -> float:
        diff = new_rot - self._prev_rotation
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        smoothed = self._prev_rotation + diff * self.smooth_alpha
        self._prev_rotation = smoothed
        return float(smoothed)

    def _smooth_offset(self, new_offset: np.ndarray) -> np.ndarray:
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
        diagonal = int(np.sqrt(h**2 + w**2))
        pad_h = (diagonal - h) // 2
        pad_w = (diagonal - w) // 2

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

        src_pts = self._build_shirt_control_points(landmarks, scale, sw, sh)
        dst_pts = self._build_body_control_points(pose, landmarks, scale, rotation, sw, sh)

        if src_pts is None or dst_pts is None or len(src_pts) < 4:
            return shirt

        # Match src/dst length
        n = min(len(src_pts), len(dst_pts))
        src_pts = src_pts[:n]
        dst_pts = dst_pts[:n]

        # Reject degenerate sets where displacement is huge (numerical safety)
        displacements = dst_pts - src_pts
        max_disp = max(sh, sw) * 0.5
        valid = np.all(np.abs(displacements) < max_disp, axis=1)
        if valid.sum() < 4:
            return shirt
        src_pts = src_pts[valid]
        displacements = displacements[valid]

        try:
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

            step = 4
            ys = np.arange(0, sh, step)
            xs = np.arange(0, sw, step)
            grid_x, grid_y = np.meshgrid(xs, ys)
            query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

            dx = rbf_x(query_pts).reshape(len(ys), len(xs))
            dy = rbf_y(query_pts).reshape(len(ys), len(xs))

            map_x_small = grid_x.astype(np.float32) + dx.astype(np.float32)
            map_y_small = grid_y.astype(np.float32) + dy.astype(np.float32)

            map_x = cv2.resize(map_x_small, (sw, sh), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y_small, (sw, sh), interpolation=cv2.INTER_LINEAR)

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
        sw: int,
        sh: int,
    ) -> Optional[np.ndarray]:
        """
        Target positions in SHIRT IMAGE SPACE where each landmark should land.
        
        KEY FIX: we work in shirt-image space, not frame space.
        The rigid transform (scale+translate) already moves the shirt to roughly 
        the right place; TPS then nudges individual control points to match body
        proportions without moving the whole shirt.
        """
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip
        lm = landmarks

        if not ls or not rs or not ls.valid or not rs.valid:
            return None

        orig_scale_x = sw / max(lm.width, 1)
        orig_scale_y = sh / max(lm.height, 1)

        def scaled_pt(p):
            return np.array([p[0] * orig_scale_x, p[1] * orig_scale_y], dtype=np.float32)

        # Body measurements (frame space)
        body_sw = pose.shoulder_width
        shirt_sw = float(lm.shoulder_right[0] - lm.shoulder_left[0]) * orig_scale_x
        sw_factor = np.clip(body_sw / max(shirt_sw, 1), 0.75, 1.35)

        body_th = pose.torso_height
        shirt_th = float(lm.hem_center[1] - lm.collar_center[1]) * orig_scale_y
        th_factor = np.clip(body_th / max(shirt_th, 1), 0.6, 1.4) if body_th > 20 else 1.0

        # Collar stays fixed (rigid already placed it)
        collar_c = scaled_pt(lm.collar_center)
        collar_l = scaled_pt(lm.collar_left)
        collar_r = scaled_pt(lm.collar_right)
        collar_y = collar_c[1]

        # Shoulders — adjust width
        sh_l = scaled_pt(lm.shoulder_left)
        sh_r = scaled_pt(lm.shoulder_right)
        sh_mid = (sh_l + sh_r) / 2
        sh_l_adj = sh_mid + (sh_l - sh_mid) * sw_factor
        sh_r_adj = sh_mid + (sh_r - sh_mid) * sw_factor

        # Chest / waist / hem — stretch vertically
        def stretch_y(pt):
            dy = pt[1] - collar_y
            return np.array([pt[0], collar_y + dy * th_factor], dtype=np.float32)

        chest_adj  = stretch_y(scaled_pt(lm.chest_center))
        waist_adj  = stretch_y(scaled_pt(lm.waist_center))
        hem_adj    = stretch_y(scaled_pt(lm.hem_center))

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
        """Deform sleeve areas to follow arm positions."""
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        le = pose.left_elbow
        re = pose.right_elbow

        if not any([ls and ls.valid, rs and rs.valid]):
            return shirt

        result = shirt.copy()
        sh, sw = shirt.shape[:2]
        sx = sw / max(landmarks.width, 1)
        sy = sh / max(landmarks.height, 1)

        if ls and ls.valid and le and le.valid:
            result = self._warp_sleeve_region(
                result, landmarks, side="left",
                shoulder_kp=ls, elbow_kp=le,
                sx=sx, sy=sy, scale=scale,
            )

        if rs and rs.valid and re and re.valid:
            result = self._warp_sleeve_region(
                result, landmarks, side="right",
                shoulder_kp=rs, elbow_kp=re,
                sx=sx, sy=sy, scale=scale,
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

        src_shoulder = np.array([garment_shoulder[0] * sx, garment_shoulder[1] * sy], dtype=np.float32)
        src_sleeve   = np.array([garment_sleeve_end[0] * sx, garment_sleeve_end[1] * sy], dtype=np.float32)
        src_mid = (src_shoulder + src_sleeve) / 2

        sh_pos = shoulder_kp.to_array()
        el_pos = elbow_kp.to_array()
        arm_vec = el_pos - sh_pos
        arm_len = np.linalg.norm(arm_vec)
        if arm_len < 5:
            return shirt

        arm_dir = arm_vec / arm_len
        sleeve_len = np.linalg.norm(src_sleeve - src_shoulder)
        target_sleeve = src_shoulder + arm_dir * sleeve_len
        target_mid = (src_shoulder + target_sleeve) / 2

        src_tri = np.array([src_shoulder, src_sleeve, src_mid], dtype=np.float32)
        dst_tri = np.array([src_shoulder, target_sleeve, target_mid], dtype=np.float32)

        M = cv2.getAffineTransform(src_tri, dst_tri)

        x_min = int(min(src_shoulder[0], src_sleeve[0], target_sleeve[0])) - 15
        x_max = int(max(src_shoulder[0], src_sleeve[0], target_sleeve[0])) + 15
        y_min = int(min(src_shoulder[1], src_sleeve[1], target_sleeve[1])) - 15
        y_max = int(max(src_shoulder[1], src_sleeve[1], target_sleeve[1])) + 15
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(sw, x_max); y_max = min(sh, y_max)

        if x_max <= x_min or y_max <= y_min:
            return shirt

        roi = shirt[y_min:y_max, x_min:x_max].copy()
        M_roi = M.copy()
        M_roi[0, 2] -= x_min
        M_roi[1, 2] -= y_min

        roi_h, roi_w = roi.shape[:2]
        warped_roi = cv2.warpAffine(
            roi, M_roi, (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        result = shirt.copy()
        blend_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
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
        self._prev_scale = 1.0
        self._prev_rotation = 0.0
        self._prev_offset = np.array([0.0, 0.0])
        self._prev_ctrl_pts = None
        self._velocity_offset = np.zeros(2)
        self._velocity_scale = 0.0
        self._last_warp = None


__all__ = ["HybridWarper", "WarpResult"]