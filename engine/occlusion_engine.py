"""
occlusion_engine.py - Natural Body Part Occlusion System  [FIXED]
Ensures arms, hands, neck and head appear in front of the shirt
using layered compositing and alpha masking.

FIXES (2026-04-24):
- shirt_region mask no longer cuts holes: we use a UNION of parsing-torso
  and densepose-torso, heavily dilated, so the shirt always has a full mask.
- Import uses engine.coreutils (consistent with the rest of the project).
- composite() no longer floors shirt_alpha to 0 outside torso — instead it
  blends down gradually so edges are soft, not hard-cut.
- Arm/head occlusion masks are properly feathered before compositing.
"""

import cv2
import numpy as np
from typing import Optional, Dict

from engine.coreutils import setup_logger, PoseKeypoints, feather_mask
from engine.parsing_engine import ParsedRegions
from engine.densepose_engine import TorsoMap

logger = setup_logger("occlusion")


class OcclusionEngine:
    """
    Natural occlusion compositing for virtual try-on.

    Layering order:
        Background → Shirt (on torso) → Arms → Neck/Face/Hair
    """

    def __init__(
        self,
        feather_radius: int = 14,
        skin_detection: bool = True,
    ):
        self.feather_radius = feather_radius
        self.skin_detection = skin_detection

        # HSV skin bands — covers fair through dark skin tones
        self._skin_lower  = np.array([0,   20,  50],  dtype=np.uint8)
        self._skin_upper  = np.array([22, 255, 255],  dtype=np.uint8)
        self._skin_lower2 = np.array([155, 20,  50],  dtype=np.uint8)
        self._skin_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        self._skin_lower3 = np.array([5,   25,  30],  dtype=np.uint8)
        self._skin_upper3 = np.array([18, 200, 180],  dtype=np.uint8)

    def _ensure_mask(self, mask, h: int, w: int) -> np.ndarray:
        if mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        elif m.ndim != 2:
            return np.zeros((h, w), dtype=np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.clip(m, 0, 255).astype(np.uint8)

    def _largest_component(self, mask: np.ndarray, min_area: int = 0) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return (binary * 255).astype(np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        if int(stats[best, cv2.CC_STAT_AREA]) < max(1, int(min_area)):
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
        h, w = frame.shape[:2]

        # Normalise all external masks first
        parsed.torso     = self._ensure_mask(parsed.torso, h, w)
        parsed.left_arm  = self._ensure_mask(parsed.left_arm, h, w)
        parsed.right_arm = self._ensure_mask(parsed.right_arm, h, w)
        parsed.face      = self._ensure_mask(parsed.face, h, w)
        parsed.hair      = self._ensure_mask(parsed.hair, h, w)
        parsed.legs      = self._ensure_mask(parsed.legs, h, w)
        torso_map.torso_mask = self._ensure_mask(torso_map.torso_mask, h, w)
        torso_map.neck_mask  = self._ensure_mask(torso_map.neck_mask, h, w)
        if torso_map.arm_masks:
            torso_map.arm_masks = {
                k: self._ensure_mask(v, h, w)
                for k, v in torso_map.arm_masks.items()
            }

        masks = {}

        # ── Shirt region ──────────────────────────────────────────────────────
        shirt_region = self._compute_shirt_region(torso_map, parsed, pose, h, w)
        masks["shirt_region"] = shirt_region

        # ── Arm occlusion (arms appear IN FRONT of shirt) ─────────────────────
        arm_mask = self._compute_arm_occlusion(frame, parsed, pose, h, w)
        masks["arm_occlusion"] = arm_mask

        # ── Head/neck occlusion (head appears IN FRONT of shirt) ──────────────
        head_mask = self._compute_head_occlusion(parsed, pose, h, w)
        masks["head_occlusion"] = head_mask

        # ── Combined foreground ───────────────────────────────────────────────
        arm_mask  = self._ensure_mask(arm_mask, h, w)
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
        FIX: Build a GENEROUS shirt region so the shirt is never cut with holes.
        We UNION all available torso estimates, dilate heavily, then only subtract
        the clearly-non-body regions (face/hair/background far from torso).
        """
        shirt_region = np.zeros((h, w), dtype=np.uint8)

        # 1) Collect every torso signal we have
        sources = []
        if np.any(parsed.torso > 0):
            sources.append(parsed.torso)
        if np.any(torso_map.torso_mask > 0):
            sources.append(torso_map.torso_mask)

        if sources:
            for src in sources:
                shirt_region = cv2.bitwise_or(shirt_region, src)
        else:
            # Hard geometric fallback
            shirt_region = self._geometric_shirt_region(pose, h, w)

        # 2) Dilate generously — fills small gaps from parsing noise
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        shirt_region = cv2.dilate(shirt_region, k_big, iterations=1)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        shirt_region = cv2.morphologyEx(shirt_region, cv2.MORPH_CLOSE, k_close)

        # 3) Add arm/sleeve areas within the expanded torso zone
        sleeve_zone = cv2.dilate(
            shirt_region,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
            iterations=1,
        )
        for arm in [parsed.left_arm, parsed.right_arm]:
            if np.any(arm > 0):
                shirt_region = cv2.bitwise_or(shirt_region, cv2.bitwise_and(arm, sleeve_zone))
        if torso_map.arm_masks:
            for am in torso_map.arm_masks.values():
                if np.any(am > 0):
                    shirt_region = cv2.bitwise_or(shirt_region, cv2.bitwise_and(am, sleeve_zone))

        # 4) Only subtract HEAD (face + hair, with extra dilation so collar gap is clean)
        face = self._ensure_mask(parsed.face, h, w)
        hair = self._ensure_mask(parsed.hair, h, w)
        head = cv2.bitwise_or(face, hair)
        if np.any(head > 0):
            head_exp = cv2.dilate(
                head,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            shirt_region = cv2.bitwise_and(shirt_region, cv2.bitwise_not(head_exp))

        # 5) Keep only the largest blob (remove isolated noise patches)
        min_area = int(0.005 * h * w)
        shirt_region = self._largest_component(shirt_region, min_area=min_area)

        # 6) Final gentle smooth
        shirt_region = cv2.GaussianBlur(shirt_region, (9, 9), 0)
        _, shirt_region = cv2.threshold(shirt_region, 30, 255, cv2.THRESH_BINARY)

        return shirt_region

    def _geometric_shirt_region(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        """Pure-keypoint torso polygon — used only when all parsers fail."""
        mask = np.zeros((h, w), dtype=np.uint8)
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip

        if not (ls and rs and ls.valid and rs.valid):
            return mask

        sw = pose.shoulder_width
        expand = sw * 0.18
        pts = [
            [ls.x - expand, ls.y],
            [rs.x + expand, rs.y],
        ]
        if rh and rh.valid:
            pts.append([rh.x + expand * 0.5, rh.y + sw * 0.1])
        else:
            pts.append([rs.x, rs.y + sw * 1.5])
        if lh and lh.valid:
            pts.append([lh.x - expand * 0.5, lh.y + sw * 0.1])
        else:
            pts.append([ls.x, ls.y + sw * 1.5])

        cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _compute_arm_occlusion(
        self,
        frame: np.ndarray,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Arms that should appear in front of the shirt."""
        arm_mask = np.zeros((h, w), dtype=np.uint8)

        # Prefer parsing model
        if np.any(parsed.left_arm > 0) or np.any(parsed.right_arm > 0):
            arm_mask = cv2.bitwise_or(parsed.left_arm, parsed.right_arm)
        else:
            arm_mask = self._geometric_arm_mask(pose, h, w)

        # Add skin-detected pixels near arm keypoints for accuracy
        if self.skin_detection:
            skin = self._detect_skin(frame)
            # Intersect skin with a dilated keypoint arm mask
            geo_arm = self._geometric_arm_mask(pose, h, w)
            geo_arm_exp = cv2.dilate(
                geo_arm,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                iterations=1,
            )
            skin_arm = cv2.bitwise_and(skin, geo_arm_exp)
            arm_mask = cv2.bitwise_or(arm_mask, skin_arm)

        # Add wrist/hand regions
        hand_mask = self._compute_hand_region(pose, h, w)
        arm_mask = cv2.bitwise_or(arm_mask, hand_mask)

        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, k)
        arm_mask = feather_mask(arm_mask, self.feather_radius)
        return np.clip(arm_mask, 0, 255).astype(np.uint8)

    def _compute_head_occlusion(
        self,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Head/neck region that sits above / in front of the shirt collar."""
        if np.any(parsed.face > 0) or np.any(parsed.hair > 0):
            head_mask = cv2.bitwise_or(parsed.face, parsed.hair)
        else:
            head_mask = self._geometric_head_mask(pose, h, w)

        # Include neck keypoint region
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        nose = pose.nose
        if ls and rs and ls.valid and rs.valid and nose and nose.valid:
            neck_cx = int((ls.x + rs.x) / 2)
            neck_top = int(nose.y)
            neck_bot = int((ls.y + rs.y) / 2)
            sw = max(10.0, pose.shoulder_width)
            neck_w = max(14, int(sw * 0.18))
            neck_h = max(10, abs(neck_bot - neck_top) // 2 + 12)
            cv2.ellipse(head_mask,
                        (neck_cx, (neck_top + neck_bot) // 2),
                        (neck_w, neck_h), 0, 0, 360, 255, -1)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, k)
        head_mask = feather_mask(head_mask, self.feather_radius)
        return np.clip(head_mask, 0, 255).astype(np.uint8)

    def _geometric_arm_mask(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        sw = pose.shoulder_width
        if sw < 5:
            return mask
        arm_thick = max(12, int(sw * 0.16))

        def draw_arm(p1, p2, thickness):
            if p1 and p2 and p1.valid and p2.valid:
                cv2.line(mask, p1.to_tuple(), p2.to_tuple(), 255, thickness)
                cv2.circle(mask, p1.to_tuple(), thickness // 2, 255, -1)
                cv2.circle(mask, p2.to_tuple(), thickness // 2, 255, -1)

        draw_arm(pose.left_shoulder,  pose.left_elbow,  arm_thick)
        draw_arm(pose.left_elbow,     pose.left_wrist,  int(arm_thick * 0.9))
        draw_arm(pose.right_shoulder, pose.right_elbow, arm_thick)
        draw_arm(pose.right_elbow,    pose.right_wrist, int(arm_thick * 0.9))
        return mask

    def _compute_hand_region(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        sw = pose.shoulder_width
        if sw < 5:
            return mask
        hand_r = max(14, int(sw * 0.12))
        for wrist in [pose.left_wrist, pose.right_wrist]:
            if wrist and wrist.valid:
                cv2.circle(mask, wrist.to_tuple(), hand_r, 255, -1)
        return mask

    def _geometric_head_mask(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        nose = pose.nose
        ls   = pose.left_shoulder
        rs   = pose.right_shoulder
        if not (nose and nose.valid):
            return mask
        sw = max(50.0, pose.shoulder_width)
        head_r = int(sw * 0.24)
        face_center = (int(nose.x), int(nose.y - head_r * 0.1))
        cv2.ellipse(mask, face_center, (head_r, int(head_r * 1.25)), 0, 0, 360, 255, -1)
        if ls and rs and ls.valid and rs.valid:
            neck_cx  = int((ls.x + rs.x) / 2)
            neck_top = int(nose.y + head_r * 0.6)
            neck_bot = int((ls.y + rs.y) / 2)
            neck_w   = max(10, int(sw * 0.16))
            neck_h   = max(10, abs(neck_bot - neck_top) // 2 + 6)
            cv2.ellipse(mask,
                        (neck_cx, (neck_top + neck_bot) // 2),
                        (neck_w, neck_h), 0, 0, 360, 255, -1)
        return mask

    def _detect_skin(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self._skin_lower,  self._skin_upper)
        m2 = cv2.inRange(hsv, self._skin_lower2, self._skin_upper2)
        m3 = cv2.inRange(hsv, self._skin_lower3, self._skin_upper3)
        skin = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return skin

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
        IMPROVED Composite: background → shirt (soft masked) → body parts (feathered)
        
        Key improvements:
        - Proper edge feathering on all layers
        - Smart mask constraint that follows body shape
        - Gradual transparency at edges (not hard cutoff)
        - Better foreground integration with natural blending
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        if warped_shirt is None:
            return result

        sh, sw_s = warped_shirt.shape[:2]

        # ── Clip to frame bounds ──────────────────────────────────────────
        x1 = max(0, placement_x)
        y1 = max(0, placement_y)
        x2 = min(w, placement_x + sw_s)
        y2 = min(h, placement_y + sh)

        if x2 <= x1 or y2 <= y1:
            return result

        # Source coordinates in shirt image
        sx1 = x1 - placement_x
        sy1 = y1 - placement_y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        shirt_roi = warped_shirt[sy1:sy2, sx1:sx2]
        frame_roi = result[y1:y2, x1:x2].astype(np.float32)
        
        roi_h, roi_w = shirt_roi.shape[:2]

        # ── 1. Get shirt's intrinsic alpha ────────────────────────────────
        if shirt_roi.shape[2] == 4:
            shirt_alpha = shirt_roi[:, :, 3].astype(np.float32) / 255.0
        else:
            shirt_alpha = np.ones((roi_h, roi_w), dtype=np.float32)

        # ── 2. Create body constraint mask with GRADUAL edges ─────────────
        shirt_region = occlusion_masks.get("shirt_region")
        if shirt_region is not None:
            # Ensure mask is same size and type
            shirt_region = self._ensure_mask(shirt_region, h, w)
            region_roi = shirt_region[y1:y2, x1:x2].astype(np.float32) / 255.0
            
            # Create a proper gradient at the edges
            # 1. Shrink mask slightly to create a core region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            region_core = cv2.erode(region_roi, kernel, iterations=1)
            
            # 2. Create edge gradient by subtracting core from full
            region_edge = region_roi - region_core
            
            # 3. Blur the edge for smooth transition
            region_edge = cv2.GaussianBlur(region_edge, (15, 15), 0)
            
            # 4. Reconstruct mask with soft edges
            region_soft = region_core + region_edge
            
            # Apply the constraint gently
            # Inside body: full opacity (1.0)
            # At edges: gradual fade (down to 0.3)
            # Outside body: very subtle ghost (0.15)
            body_constraint = region_soft * 0.7 + 0.15
            
            # Apply constraint to shirt alpha
            shirt_alpha = shirt_alpha * body_constraint
        
        # ── 3. Apply user opacity setting ─────────────────────────────────
        shirt_alpha = shirt_alpha * opacity
        
        # ── 4. Final edge smoothing for the complete alpha ────────────────
        # This prevents any remaining hard edges
        shirt_alpha = cv2.GaussianBlur(shirt_alpha, (7, 7), 0)
        
        # ── 5. Blend shirt with background ────────────────────────────────
        shirt_bgr = shirt_roi[:, :, :3].astype(np.float32)
        alpha_3 = np.stack([shirt_alpha] * 3, axis=2)
        
        # Alpha blending: result = bg * (1 - alpha) + shirt * alpha
        blended = frame_roi * (1.0 - alpha_3) + shirt_bgr * alpha_3
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        # ── 6. Re-composite foreground elements with FEATHERING ────────────
        foreground = occlusion_masks.get("foreground")
        if foreground is not None:
            foreground = self._ensure_mask(foreground, h, w)
            fg_roi = foreground[y1:y2, x1:x2].astype(np.float32) / 255.0
            
            # Feather the foreground mask for natural integration
            fg_roi = cv2.GaussianBlur(fg_roi, (5, 5), 0)
            
            # Create a soft inner edge for arms/head
            # This prevents the "pasted on" look
            fg_core = cv2.erode(fg_roi, np.ones((3, 3), np.float32), iterations=1)
            fg_edge = fg_roi - fg_core
            fg_edge = cv2.GaussianBlur(fg_edge, (3, 3), 0)
            fg_soft = fg_core + fg_edge * 0.8  # Slightly reduce edge opacity
            
            fg_3 = np.stack([fg_soft] * 3, axis=2)
            
            # Get original frame pixels for foreground areas
            orig_roi = frame[y1:y2, x1:x2].astype(np.float32)
            current_roi = result[y1:y2, x1:x2].astype(np.float32)
            
            # Blend: where foreground is active, show original frame
            final_roi = current_roi * (1.0 - fg_3) + orig_roi * fg_3
            result[y1:y2, x1:x2] = np.clip(final_roi, 0, 255).astype(np.uint8)

        return result


__all__ = ["OcclusionEngine"]