"""
occlusion_engine.py - Natural Body Part Occlusion System
Ensures arms, hands, neck and head appear in front of the shirt
using layered compositing and alpha masking.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict

from engine.utils import setup_logger, PoseKeypoints, feather_mask
from engine.parsing_engine import ParsedRegions
from engine.densepose_engine import TorsoMap

logger = setup_logger("occlusion")


class OcclusionEngine:
    """
    Natural occlusion compositing for virtual try-on.

    Handles layering order:
    ┌─ Background (camera frame) ──────────────────────────┐
    │ ← Shirt warped overlay (torso region)                │
    │   ← Arms (in front of shirt)                        │
    │     ← Neck + Face (above collar)                    │
    │       ← Hair (topmost)                              │
    └──────────────────────────────────────────────────────┘

    Key techniques:
    - Skin color detection for precise arm masks
    - Torso-constrained shirt rendering
    - Feathered occlusion boundaries for realistic blending
    - Hand region special handling (extended beyond sleeve)
    """

    def __init__(
        self,
        feather_radius: int = 12,
        skin_detection: bool = True,
    ):
        self.feather_radius = feather_radius
        self.skin_detection = skin_detection

        # Skin color range in HSV (broad range for diverse skin tones)
        self._skin_lower = np.array([0, 20, 60], dtype=np.uint8)
        self._skin_upper = np.array([30, 255, 255], dtype=np.uint8)
        self._skin_lower2 = np.array([160, 20, 60], dtype=np.uint8)
        self._skin_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    def _ensure_mask(self, mask: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
        """Normalize any mask to uint8 (h, w) so cv2 bitwise ops are safe."""
        if mask is None:
            return np.zeros((h, w), dtype=np.uint8)

        m = np.asarray(mask)

        # Collapse multi-channel masks to single channel.
        if m.ndim == 3:
            m = m[:, :, 0]
        elif m.ndim != 2:
            return np.zeros((h, w), dtype=np.uint8)

        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        if m.dtype != np.uint8:
            m = np.clip(m, 0, 255).astype(np.uint8)

        return m

    def _largest_component(self, mask: np.ndarray, min_area: int = 0) -> np.ndarray:
        """Keep only the largest connected component to suppress stray islands."""
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return (binary * 255).astype(np.uint8)

        # Ignore background label 0.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        best_area = int(stats[best, cv2.CC_STAT_AREA])
        if best_area < max(1, int(min_area)):
            return np.zeros_like(mask, dtype=np.uint8)

        out = np.zeros_like(mask, dtype=np.uint8)
        out[labels == best] = 255
        return out

    def build_occlusion_masks(
        self,
        frame: np.ndarray,
        pose: PoseKeypoints,
        parsed: ParsedRegions,
        torso_map: TorsoMap,
    ) -> Dict[str, np.ndarray]:
        """
        Build all occlusion masks for compositing.

        Returns dict with:
            'shirt_region': where to render shirt
            'arm_occlusion': arms that cover shirt
            'head_occlusion': head/neck/hair above collar
            'torso_constraint': prevents shirt from rendering outside torso
        """
        h, w = frame.shape[:2]
        masks = {}

        # Normalize all external masks to frame shape/type before any bitwise ops.
        parsed.torso = self._ensure_mask(parsed.torso, h, w)
        parsed.left_arm = self._ensure_mask(parsed.left_arm, h, w)
        parsed.right_arm = self._ensure_mask(parsed.right_arm, h, w)
        parsed.face = self._ensure_mask(parsed.face, h, w)
        parsed.hair = self._ensure_mask(parsed.hair, h, w)
        parsed.legs = self._ensure_mask(parsed.legs, h, w)

        torso_map.torso_mask = self._ensure_mask(torso_map.torso_mask, h, w)
        torso_map.neck_mask = self._ensure_mask(torso_map.neck_mask, h, w)
        if torso_map.arm_masks:
            torso_map.arm_masks = {
                k: self._ensure_mask(v, h, w)
                for k, v in torso_map.arm_masks.items()
            }

        # ── Shirt rendering region ────────────────────────────────
        # Shirt should appear on torso but not arms, face, background
        shirt_region = self._compute_shirt_region(
            torso_map, parsed, pose, h, w
        )
        masks["shirt_region"] = shirt_region

        # ── Arm occlusion mask ────────────────────────────────────
        arm_mask = self._compute_arm_occlusion(frame, parsed, pose, h, w)
        masks["arm_occlusion"] = arm_mask

        # ── Head occlusion mask ───────────────────────────────────
        head_mask = self._compute_head_occlusion(parsed, pose, h, w)
        masks["head_occlusion"] = head_mask

        # ── Combined foreground (body parts in front of shirt) ────
        arm_mask = self._ensure_mask(arm_mask, h, w)
        head_mask = self._ensure_mask(head_mask, h, w)
        foreground = cv2.bitwise_or(arm_mask, head_mask)
        foreground = feather_mask(foreground, self.feather_radius // 2)
        foreground = np.clip(foreground, 0, 255).astype(np.uint8)
        masks["foreground"] = foreground

        return masks

    def _compute_shirt_region(
        self,
        torso_map: TorsoMap,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Compute where shirt should be rendered.
        Combines torso map with parsing to get clean shirt region.
        """
        shirt_region = np.zeros((h, w), dtype=np.uint8)

        # Prefer parsing model result if available
        if parsed.method in ("model", "parsing") and np.any(parsed.torso > 0):
            shirt_region = parsed.torso.copy()
        elif np.any(torso_map.torso_mask > 0):
            shirt_region = torso_map.torso_mask.copy()
        else:
            # Full fallback: use entire frame minus background
            shirt_region = self._geometric_shirt_region(pose, h, w)

        # Expand slightly to avoid gaps at seams
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        shirt_region = cv2.dilate(shirt_region, kernel, iterations=1)

        # Include arm regions for sleeve rendering, but constrain to torso vicinity
        # so lifted hands/background fragments don't receive shirt pixels.
        sleeve_zone = cv2.dilate(
            shirt_region,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
            iterations=1,
        )
        if np.any(parsed.left_arm > 0):
            arm = cv2.bitwise_and(parsed.left_arm, sleeve_zone)
            shirt_region = cv2.bitwise_or(shirt_region, arm)
        if np.any(parsed.right_arm > 0):
            arm = cv2.bitwise_and(parsed.right_arm, sleeve_zone)
            shirt_region = cv2.bitwise_or(shirt_region, arm)
        if torso_map.arm_masks:
            for arm_mask in torso_map.arm_masks.values():
                if np.any(arm_mask > 0):
                    arm = cv2.bitwise_and(arm_mask, sleeve_zone)
                    shirt_region = cv2.bitwise_or(shirt_region, arm)

        # Remove face/head region from shirt area
        # NOTE: Normalize face and hair to (h, w) before calling the property
        # to avoid cv2.bitwise_or size mismatch inside head_region.
        parsed.face = self._ensure_mask(parsed.face, h, w)
        parsed.hair = self._ensure_mask(parsed.hair, h, w)
        head_region = cv2.bitwise_or(parsed.face, parsed.hair)
        if np.any(head_region > 0):
            shirt_region = cv2.bitwise_and(
                shirt_region,
                cv2.bitwise_not(head_region)
            )

        # Suppress tiny disconnected patches from noisy parsing output.
        min_area = int(0.01 * h * w)
        cleaned = self._largest_component(shirt_region, min_area=min_area)
        if np.count_nonzero(cleaned) > 0:
            shirt_region = cleaned

        return shirt_region

    def _compute_arm_occlusion(
        self,
        frame: np.ndarray,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Compute arm occlusion mask - where arms appear in FRONT of shirt.
        Uses parsing mask + optional skin color detection for precision.
        """
        arm_mask = np.zeros((h, w), dtype=np.uint8)

        # Start with parsing model arms
        # NOTE: Normalize both arm masks to (h, w) before combining to avoid
        # cv2.bitwise_or size mismatch when masks come from 512x512 parsing space.
        parsed.left_arm = self._ensure_mask(parsed.left_arm, h, w)
        parsed.right_arm = self._ensure_mask(parsed.right_arm, h, w)
        if np.any(parsed.left_arm > 0) or np.any(parsed.right_arm > 0):
            arm_mask = cv2.bitwise_or(parsed.left_arm, parsed.right_arm)

        # Augment with geometric arm positions
        geo_arm = self._geometric_arm_mask(pose, h, w)
        arm_mask = cv2.bitwise_or(arm_mask, geo_arm)

        # Refine with skin color detection
        if self.skin_detection and np.any(arm_mask > 0):
            skin_mask = self._detect_skin(frame)
            # Keep arm mask only where skin is detected
            arm_mask = cv2.bitwise_and(arm_mask, skin_mask)
            # Fill holes from skin detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, kernel)

        # Add hand regions (always in front)
        hand_mask = self._compute_hand_region(pose, h, w)
        arm_mask = cv2.bitwise_or(arm_mask, hand_mask)

        # Feather for smooth blending
        arm_mask = feather_mask(arm_mask, self.feather_radius)
        arm_mask = np.clip(arm_mask * 1.3, 0, 255).astype(np.uint8)

        return arm_mask

    def _compute_head_occlusion(
        self,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Compute head/neck occlusion - parts that appear above collar.
        """
        head_mask = np.zeros((h, w), dtype=np.uint8)

        # Use parsing regions
        head_region = parsed.head_region
        if np.any(head_region > 0):
            head_mask = head_region.copy()

        # Augment with geometric head region
        geo_head = self._geometric_head_mask(pose, h, w)
        head_mask = cv2.bitwise_or(head_mask, geo_head)

        # Feather
        head_mask = feather_mask(head_mask, self.feather_radius // 2)
        head_mask = np.clip(head_mask, 0, 255).astype(np.uint8)

        return head_mask

    def _geometric_shirt_region(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Fallback geometric shirt region from keypoints."""
        mask = np.zeros((h, w), dtype=np.uint8)
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip

        if not (ls and rs and ls.valid and rs.valid):
            return mask

        sw = pose.shoulder_width
        expand = sw * 0.15

        pts = []
        pts.append([ls.x - expand, ls.y])
        pts.append([rs.x + expand, rs.y])

        if rh and rh.valid:
            pts.append([rh.x + expand * 0.5, rh.y + sw * 0.1])
        else:
            pts.append([rs.x, rs.y + sw * 1.5])

        if lh and lh.valid:
            pts.append([lh.x - expand * 0.5, lh.y + sw * 0.1])
        else:
            pts.append([ls.x, ls.y + sw * 1.5])

        cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _geometric_arm_mask(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Build geometric arm mask from keypoints."""
        mask = np.zeros((h, w), dtype=np.uint8)
        sw = pose.shoulder_width
        if sw < 5:
            return mask

        arm_thick = max(10, int(sw * 0.14))

        def draw_arm_segment(p1, p2, thickness):
            if p1 and p2 and p1.valid and p2.valid:
                cv2.line(mask, p1.to_tuple(), p2.to_tuple(), 255, thickness)
                cv2.circle(mask, p1.to_tuple(), thickness // 2, 255, -1)
                cv2.circle(mask, p2.to_tuple(), thickness // 2, 255, -1)

        # Left arm
        draw_arm_segment(pose.left_shoulder, pose.left_elbow, arm_thick)
        draw_arm_segment(pose.left_elbow, pose.left_wrist, int(arm_thick * 0.9))

        # Right arm
        draw_arm_segment(pose.right_shoulder, pose.right_elbow, arm_thick)
        draw_arm_segment(pose.right_elbow, pose.right_wrist, int(arm_thick * 0.9))

        return mask

    def _compute_hand_region(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Compute hand/wrist region mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        sw = pose.shoulder_width
        if sw < 5:
            return mask

        hand_r = max(12, int(sw * 0.1))

        for wrist in [pose.left_wrist, pose.right_wrist]:
            if wrist and wrist.valid:
                cv2.circle(mask, wrist.to_tuple(), hand_r, 255, -1)

        return mask

    def _geometric_head_mask(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Geometric head/neck mask from nose/shoulder keypoints."""
        mask = np.zeros((h, w), dtype=np.uint8)
        nose = pose.nose
        ls = pose.left_shoulder
        rs = pose.right_shoulder

        if not (nose and nose.valid):
            return mask

        sw = pose.shoulder_width if pose.shoulder_width > 5 else 100

        # Head ellipse
        head_r = int(sw * 0.22)
        face_center = (int(nose.x), int(nose.y - head_r * 0.1))
        cv2.ellipse(mask, face_center, (head_r, int(head_r * 1.2)), 0, 0, 360, 255, -1)

        # Neck
        if ls and rs and ls.valid and rs.valid:
            neck_cx = int((ls.x + rs.x) / 2)
            neck_top = int(nose.y + head_r * 0.6)
            neck_bot = int((ls.y + rs.y) / 2)
            neck_w = max(8, int(sw * 0.15))
            neck_h = max(8, abs(neck_bot - neck_top) // 2)
            cv2.ellipse(
                mask,
                (neck_cx, (neck_top + neck_bot) // 2),
                (neck_w, neck_h),
                0, 0, 360, 255, -1,
            )

        return mask

    def _detect_skin(self, frame: np.ndarray) -> np.ndarray:
        """Detect skin pixels using HSV color model."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self._skin_lower, self._skin_upper)
        mask2 = cv2.inRange(hsv, self._skin_lower2, self._skin_upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        return skin_mask

    def composite(
        self,
        frame: np.ndarray,
        warped_shirt: np.ndarray,
        placement_x: int,
        placement_y: int,
        occlusion_masks: Dict[str, np.ndarray],
        opacity: float = 0.95,
    ) -> np.ndarray:
        """
        Final compositing: background → shirt → foreground body parts.

        Args:
            frame: Original camera frame (BGR)
            warped_shirt: Warped shirt overlay (BGRA)
            placement_x, placement_y: Shirt placement offset
            occlusion_masks: Output from build_occlusion_masks()
            opacity: Overall shirt opacity

        Returns:
            Final composited BGR frame
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        if warped_shirt is None:
            return result

        sh, sw = warped_shirt.shape[:2]

        # Clip to frame bounds
        x1 = max(0, placement_x)
        y1 = max(0, placement_y)
        x2 = min(w, placement_x + sw)
        y2 = min(h, placement_y + sh)

        if x2 <= x1 or y2 <= y1:
            return result

        # Shirt ROI
        sx1 = x1 - placement_x
        sy1 = y1 - placement_y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        shirt_roi = warped_shirt[sy1:sy2, sx1:sx2]
        frame_roi = result[y1:y2, x1:x2].astype(np.float32)

        # Get shirt alpha
        if shirt_roi.shape[2] == 4:
            shirt_alpha = shirt_roi[:, :, 3].astype(np.float32) / 255.0
        else:
            shirt_alpha = np.ones((sy2 - sy1, sx2 - sx1), dtype=np.float32)

        shirt_alpha = shirt_alpha * opacity

        # Apply shirt region constraint (don't render shirt outside torso area)
        shirt_region = occlusion_masks.get("shirt_region")
        if shirt_region is not None:
            shirt_region = self._ensure_mask(shirt_region, h, w)
            region_roi = shirt_region[y1:y2, x1:x2].astype(np.float32) / 255.0
            region_roi = cv2.GaussianBlur(region_roi, (7, 7), 0)
            region_roi = np.where(region_roi > 0.08, region_roi, 0.0)
            shirt_alpha = shirt_alpha * region_roi
            shirt_alpha = np.where(shirt_alpha > 0.03, shirt_alpha, 0.0)

        # Blend shirt over background
        shirt_bgr = shirt_roi[:, :, :3].astype(np.float32)
        alpha_3 = np.stack([shirt_alpha] * 3, axis=2)
        blended_with_shirt = frame_roi * (1 - alpha_3) + shirt_bgr * alpha_3

        result[y1:y2, x1:x2] = np.clip(blended_with_shirt, 0, 255).astype(np.uint8)

        # ── Re-composite foreground body parts OVER shirt ─────────
        foreground = occlusion_masks.get("foreground")
        if foreground is not None:
            foreground = self._ensure_mask(foreground, h, w)
            fg_roi = foreground[y1:y2, x1:x2].astype(np.float32) / 255.0
            fg_roi = np.clip(fg_roi, 0, 1)
            fg_alpha_3 = np.stack([fg_roi] * 3, axis=2)

            # Original frame pixels at foreground regions
            orig_roi = frame[y1:y2, x1:x2].astype(np.float32)
            current_roi = result[y1:y2, x1:x2].astype(np.float32)

            # Blend: show original skin/hair over shirt
            final_roi = current_roi * (1 - fg_alpha_3) + orig_roi * fg_alpha_3
            result[y1:y2, x1:x2] = np.clip(final_roi, 0, 255).astype(np.uint8)

        return result


__all__ = ["OcclusionEngine"]
