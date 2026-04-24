"""
hybrid_warper.py - FULL WARPER V2
Professional body-fit virtual try-on warper.

UPGRADES:
- True shoulder anchoring
- Better torso scaling
- Smart neck alignment
- Strong TPS wrapping
- Sleeve follow
- Stable smoothing
- No floating shirt issue
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.interpolate import RBFInterpolator

from engine.coreutils import (
    setup_logger,
    PoseKeypoints,
    smooth_array,
    smooth_value,
)
from engine.garment_landmarks import GarmentLandmarks

logger = setup_logger("hybrid_warper")


@dataclass
class WarpResult:
    warped_shirt: np.ndarray
    placement_x: int
    placement_y: int
    scale: float
    rotation: float
    target_width: int
    target_height: int
    confidence: float = 1.0


class HybridWarper:

    def __init__(
        self,
        smooth_alpha: float = 0.38,
        physics_lag: float = 0.18,
        max_scale: float = 3.0,
        min_scale: float = 0.35,
        tps_smooth: float = 0.08,
    ):
        self.smooth_alpha = smooth_alpha
        self.physics_lag = physics_lag
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.tps_smooth = tps_smooth

        self._prev_scale = 1.0
        self._prev_rot = 0.0
        self._prev_offset = np.array([0.0, 0.0], dtype=np.float32)
        self._vel = np.array([0.0, 0.0], dtype=np.float32)
        self._prev_profile: Optional[Dict[str, float]] = None

    # ==========================================================
    # MAIN
    # ==========================================================
    def warp(
        self,
        shirt_image: np.ndarray,
        landmarks: GarmentLandmarks,
        pose: PoseKeypoints,
        frame_shape: Tuple[int, int],
        torso_mask: Optional[np.ndarray] = None,
    ) -> Optional[WarpResult]:

        if pose is None or not pose.is_usable(min_keypoints=4):
            return None

        fh, fw = frame_shape[:2]

        scale, rot, offset = self._compute_transform(
            landmarks, pose, fw, fh, torso_mask=torso_mask
        )

        scale = self._smooth_scale(scale)
        rot = self._smooth_rot(rot)
        offset = self._smooth_offset(offset)

        sh, sw = shirt_image.shape[:2]

        tw = int(sw * scale)
        th = int(sh * scale)

        if tw < 10 or th < 10:
            return None

        shirt = cv2.resize(
            shirt_image,
            (tw, th),
            interpolation=cv2.INTER_LANCZOS4
        )

        if abs(rot) > 0.4:
            shirt = self._rotate_image(shirt, rot)

        shirt = self._apply_tps(
            shirt,
            landmarks,
            pose,
            torso_mask=torso_mask,
        )

        shirt = self._sleeve_follow(
            shirt,
            landmarks,
            pose
        )

        return WarpResult(
            warped_shirt=shirt,
            placement_x=int(offset[0]),
            placement_y=int(offset[1]),
            scale=scale,
            rotation=rot,
            target_width=shirt.shape[1],
            target_height=shirt.shape[0],
        )

    # ==========================================================
    # TRANSFORM
    # ==========================================================
    def _compute_transform(self, lm, pose, fw, fh, torso_mask=None):

        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip
        nose = pose.nose

        if not ls or not rs:
            return 1.0, 0.0, np.array([fw * 0.3, fh * 0.2])

        body_sw = max(10.0, pose.shoulder_width)
        torso_bbox = self._mask_bbox(torso_mask)

        shirt_sw = max(
            20,
            float(lm.shoulder_right[0] - lm.shoulder_left[0])
        )

        if torso_bbox is not None:
            _, _, box_w, box_h = torso_bbox
            body_sw = body_sw * 0.70 + float(max(10, box_w)) * 0.30

        scale = (body_sw / shirt_sw) * 1.20

        if torso_bbox is not None:
            _, _, _, box_h = torso_bbox
            shirt_h = max(30, float(lm.hem_center[1] - lm.collar_center[1]))
            h_scale = (float(max(10, box_h)) / shirt_h) * 1.08
            scale = scale * 0.62 + h_scale * 0.38
        elif lh and rh and lh.valid and rh.valid:
            body_h = pose.torso_height
            shirt_h = max(
                30,
                float(lm.hem_center[1] - lm.collar_center[1])
            )
            h_scale = body_h / shirt_h
            scale = scale * 0.72 + h_scale * 0.28

        scale = float(np.clip(scale, self.min_scale, self.max_scale))

        rot = float(((pose.torso_angle + 90.0) % 180.0) - 90.0)
        rot = float(np.clip(rot, -35.0, 35.0))

        mid_x = (ls.x + rs.x) / 2
        mid_y = (ls.y + rs.y) / 2

        if torso_bbox is not None:
            bx, by, bw, bh = torso_bbox
            mid_x = mid_x * 0.65 + (bx + bw * 0.5) * 0.35
            if nose and nose.valid:
                neck_y = (nose.y + mid_y) / 2
            else:
                neck_y = by + bh * 0.11
        elif nose and nose.valid:
            neck_y = (nose.y + mid_y) / 2
        else:
            neck_y = mid_y - body_sw * 0.20

        collar = np.array(lm.collar_center, dtype=np.float32) * scale

        offset = np.array([
            mid_x - collar[0],
            neck_y - collar[1]
        ], dtype=np.float32)

        return scale, rot, offset

    # ==========================================================
    # TPS BODY WRAP
    # ==========================================================
    def _apply_tps(self, img, lm, pose, torso_mask=None):

        h, w = img.shape[:2]

        src = np.array([
            lm.collar_center,
            lm.shoulder_left,
            lm.shoulder_right,
            lm.chest_left,
            lm.chest_right,
            lm.waist_left,
            lm.waist_right,
            lm.hem_left,
            lm.hem_right,
            lm.hem_center
        ], dtype=np.float32)

        sx = w / max(lm.width, 1)
        sy = h / max(lm.height, 1)

        src[:, 0] *= sx
        src[:, 1] *= sy

        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip

        if not ls or not rs:
            return img

        dst = src.copy()
        shirt_sw = float(max(10.0, src[2][0] - src[1][0]))
        shirt_cx = float((src[1][0] + src[2][0]) * 0.5)

        profile = self._smooth_profile(self._torso_profile(torso_mask))
        if profile is not None:
            shoulder_w = float(max(10.0, profile["shoulder_w"]))
            shoulder_cx = float(profile["shoulder_cx"])
            shift_ratio = float(np.clip((shoulder_cx - 0.5), -0.08, 0.08))
            center_x = shirt_cx + shift_ratio * shirt_sw

            chest_ratio_raw = float(np.clip(profile["chest_w"] / shoulder_w, 0.92, 1.06))
            waist_ratio_raw = float(np.clip(profile["waist_w"] / shoulder_w, 0.92, 1.05))
            hem_ratio_raw = float(np.clip(profile["hem_w"] / shoulder_w, 0.92, 1.08))

            # Keep TPS width deformation conservative to avoid over-pinched waists.
            blend = 0.18
            chest_ratio = 1.0 + (chest_ratio_raw - 1.0) * blend
            waist_ratio = 1.0 + (waist_ratio_raw - 1.0) * blend
            hem_ratio = 1.0 + (hem_ratio_raw - 1.0) * blend

            chest_w = shirt_sw * chest_ratio
            waist_w = shirt_sw * waist_ratio
            hem_w = shirt_sw * hem_ratio

            # Prevent non-physical pinch by preserving most of source silhouette width.
            src_chest_w = float(max(8.0, src[4][0] - src[3][0]))
            src_waist_w = float(max(8.0, src[6][0] - src[5][0]))
            src_hem_w = float(max(8.0, src[8][0] - src[7][0]))
            chest_w = max(chest_w, src_chest_w * 0.96)
            waist_w = max(waist_w, src_waist_w * 0.96)
            hem_w = max(hem_w, src_hem_w * 0.96)

            dst[3][0] = center_x - chest_w * 0.5
            dst[4][0] = center_x + chest_w * 0.5
            dst[5][0] = center_x - waist_w * 0.5
            dst[6][0] = center_x + waist_w * 0.5
            dst[7][0] = center_x - hem_w * 0.5
            dst[8][0] = center_x + hem_w * 0.5
            dst[9][0] = center_x

            chest_y = float(np.clip(profile["chest_y_rel"] * h, 0.0, h - 1.0))
            waist_y = float(np.clip(profile["waist_y_rel"] * h, 0.0, h - 1.0))
            hem_y = float(np.clip(profile["hem_y_rel"] * h, 0.0, h - 1.0))
            dst[3][1] = src[3][1] * 0.82 + chest_y * 0.18
            dst[4][1] = src[4][1] * 0.82 + chest_y * 0.18
            dst[5][1] = src[5][1] * 0.80 + waist_y * 0.20
            dst[6][1] = src[6][1] * 0.80 + waist_y * 0.20
            dst[7][1] = src[7][1] * 0.76 + hem_y * 0.24
            dst[8][1] = src[8][1] * 0.76 + hem_y * 0.24
            dst[9][1] = src[9][1] * 0.72 + hem_y * 0.28

        # Keep deformation smooth and bounded.
        dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
        dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)

        try:
            grid_x, grid_y = np.meshgrid(
                np.arange(w),
                np.arange(h)
            )

            pts = np.stack(
                [grid_x.ravel(), grid_y.ravel()],
                axis=-1
            )

            rbf_x = RBFInterpolator(
                src, dst[:, 0],
                kernel="thin_plate_spline",
                smoothing=self.tps_smooth
            )

            rbf_y = RBFInterpolator(
                src, dst[:, 1],
                kernel="thin_plate_spline",
                smoothing=self.tps_smooth
            )

            map_x = rbf_x(pts).reshape(h, w).astype(np.float32)
            map_y = rbf_y(pts).reshape(h, w).astype(np.float32)

            warped = cv2.remap(
                img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )

            return warped

        except Exception:
            return img

    # ==========================================================
    # SLEEVE FOLLOW
    # ==========================================================
    def _sleeve_follow(self, img, lm, pose):
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            return img

        sx = w / max(1.0, float(lm.width))
        sy = h / max(1.0, float(lm.height))

        left_shoulder = np.array(lm.shoulder_left, dtype=np.float32) * np.array([sx, sy], dtype=np.float32)
        right_shoulder = np.array(lm.shoulder_right, dtype=np.float32) * np.array([sx, sy], dtype=np.float32)
        left_sleeve = np.array(lm.sleeve_left_end, dtype=np.float32) * np.array([sx, sy], dtype=np.float32)
        right_sleeve = np.array(lm.sleeve_right_end, dtype=np.float32) * np.array([sx, sy], dtype=np.float32)

        out = img.copy()
        out = self._warp_single_sleeve(out, left_shoulder, left_sleeve, pose.left_shoulder, pose.left_elbow, pose.left_wrist)
        out = self._warp_single_sleeve(out, right_shoulder, right_sleeve, pose.right_shoulder, pose.right_elbow, pose.right_wrist)
        return out

    def _warp_single_sleeve(
        self,
        img: np.ndarray,
        src_shoulder: np.ndarray,
        src_sleeve: np.ndarray,
        pose_shoulder,
        pose_elbow,
        pose_wrist,
    ) -> np.ndarray:
        if pose_shoulder is None or not pose_shoulder.valid:
            return img
        anchor = pose_wrist if (pose_wrist is not None and pose_wrist.valid) else pose_elbow
        if anchor is None or not anchor.valid:
            return img

        vec = np.array([anchor.x - pose_shoulder.x, anchor.y - pose_shoulder.y], dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n < 1e-3:
            return img
        direction = vec / n

        sleeve_len = float(np.linalg.norm(src_sleeve - src_shoulder))
        if sleeve_len < 4.0:
            return img
        target_len = float(np.clip(n * 0.30, sleeve_len * 0.85, sleeve_len * 1.20))
        dst_sleeve = src_shoulder + direction * target_len

        src_mid = (src_shoulder + src_sleeve) * 0.5
        dst_mid = (src_shoulder + dst_sleeve) * 0.5

        src_tri = np.array([src_shoulder, src_sleeve, src_mid], dtype=np.float32)
        dst_tri = np.array([src_shoulder, dst_sleeve, dst_mid], dtype=np.float32)
        M = cv2.getAffineTransform(src_tri, dst_tri)

        h, w = img.shape[:2]
        warped = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        mask = np.zeros((h, w), dtype=np.float32)
        radius = int(max(10.0, sleeve_len * 0.40))
        cv2.line(mask, tuple(np.int32(src_shoulder)), tuple(np.int32(src_sleeve)), 1.0, radius)
        cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, sleeve_len * 0.10), dst=mask)
        mask = np.clip(mask, 0.0, 1.0)
        mask3 = np.repeat(mask[:, :, None], img.shape[2], axis=2)
        out = img.astype(np.float32) * (1.0 - mask3) + warped.astype(np.float32) * mask3
        return np.clip(out, 0, 255).astype(np.uint8)

    # ==========================================================
    # HELPERS
    # ==========================================================
    def _rotate_image(self, img, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        return cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0, 0, 0, 0]
        )

    def _smooth_scale(self, v):
        self._prev_scale = smooth_value(
            self._prev_scale,
            v,
            alpha=self.smooth_alpha
        )
        return self._prev_scale

    def _smooth_rot(self, v):
        self._prev_rot = smooth_value(
            self._prev_rot,
            v,
            alpha=self.smooth_alpha
        )
        return self._prev_rot

    def _smooth_offset(self, v):
        self._vel = smooth_array(
            self._vel,
            v - self._prev_offset,
            alpha=0.35
        )

        self._prev_offset = smooth_array(
            self._prev_offset,
            v,
            alpha=self.smooth_alpha
        )

        return self._prev_offset + self._vel * self.physics_lag

    def reset(self):
        self._prev_scale = 1.0
        self._prev_rot = 0.0
        self._prev_offset = np.array([0.0, 0.0], dtype=np.float32)
        self._vel = np.array([0.0, 0.0], dtype=np.float32)
        self._prev_profile = None

    def _mask_bbox(self, mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
        if mask is None or mask.size == 0:
            return None
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        ys, xs = np.where(m > 10)
        if len(xs) < 50:
            return None
        x1, x2 = int(np.min(xs)), int(np.max(xs))
        y1, y2 = int(np.min(ys)), int(np.max(ys))
        return x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)

    def _torso_profile(self, torso_mask: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        bbox = self._mask_bbox(torso_mask)
        if bbox is None:
            return None
        x, y, w, h = bbox
        if w < 10 or h < 10:
            return None
        m = np.asarray(torso_mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        roi = m[y:y + h, x:x + w]

        def row_stats(rel_y: float):
            yy = int(np.clip(rel_y * (h - 1), 0, h - 1))
            row = roi[yy] > 10
            if not np.any(row):
                return 0.0, 0.5, rel_y
            idx = np.where(row)[0]
            width = float(idx[-1] - idx[0] + 1)
            center = float((idx[-1] + idx[0]) * 0.5 / max(1, w - 1))
            return width, center, float(yy / max(1, h - 1))

        shoulder_w, shoulder_cx, shoulder_y_rel = row_stats(0.13)
        chest_w, chest_cx, chest_y_rel = row_stats(0.36)
        waist_w, waist_cx, waist_y_rel = row_stats(0.63)
        hem_w, hem_cx, hem_y_rel = row_stats(0.90)

        if shoulder_w < 10:
            return None

        return {
            "shoulder_w": shoulder_w,
            "chest_w": chest_w,
            "waist_w": waist_w,
            "hem_w": hem_w,
            "shoulder_cx": shoulder_cx,
            "chest_cx": chest_cx,
            "waist_cx": waist_cx,
            "hem_cx": hem_cx,
            "shoulder_y_rel": shoulder_y_rel,
            "chest_y_rel": chest_y_rel,
            "waist_y_rel": waist_y_rel,
            "hem_y_rel": hem_y_rel,
        }

    def _smooth_profile(self, profile: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if profile is None:
            self._prev_profile = None
            return None
        if self._prev_profile is None:
            self._prev_profile = profile.copy()
            return profile

        alpha = 0.15
        smoothed: Dict[str, float] = {}
        for k, v in profile.items():
            pv = float(self._prev_profile.get(k, v))
            smoothed[k] = pv * (1.0 - alpha) + float(v) * alpha
        self._prev_profile = smoothed
        return smoothed


__all__ = ["HybridWarper", "WarpResult"]
