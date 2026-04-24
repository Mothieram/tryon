"""
render_pipeline.py - Master Render Pipeline  [FIXED]
Orchestrates all engines to produce final try-on frames.

FIXES (2026-04-24):
- Imports corrected: engine.coreutils throughout (no more engine.utils mismatch)
- Shirt lighting now actually applied (was a dead copy)
- Shadow block enabled behind flag (was `if False`)
- _process_frame_inner now correctly passes frame shape as tuple
- GarmentMeta defaults tuned for real shirt PNGs (collar at ~8–12% from top)
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

from engine.coreutils import (
    setup_logger, PoseKeypoints, GarmentMeta,
    FPSCounter, FrameCache, ensure_bgra, create_placeholder_shirt,
    alpha_blend,
)
from engine.yolo_pose import YoloPoseEngine, AsyncPoseEngine
from engine.densepose_engine import DensePoseEngine
from engine.parsing_engine import ParsingEngine
from engine.garment_landmarks import GarmentAnalyzer, GarmentLandmarks
from engine.hybrid_warper import HybridWarper
from engine.occlusion_engine import OcclusionEngine
from engine.shadow_engine import ShadowEngine

logger = setup_logger("pipeline")


@dataclass
class PipelineStats:
    fps: float = 0.0
    pose_ms: float = 0.0
    warp_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    pose_detected: bool = False
    active_shirt: str = ""
    engine_method: str = ""
    gpu_active: bool = False


@dataclass
class GarmentEntry:
    path: str
    name: str
    image: np.ndarray           # BGRA
    meta: GarmentMeta
    landmarks: Optional[GarmentLandmarks] = None
    thumbnail: Optional[np.ndarray] = None


class RenderPipeline:
    """
    Master Virtual Try-On Render Pipeline.
    """

    def __init__(
        self,
        pose_model: str = "yolov8n-pose.pt",
        parsing_model: Optional[str] = None,
        device: str = "auto",
        target_fps: int = 30,
        enable_shadows: bool = True,
        enable_lighting: bool = True,
        opacity: float = 0.95,
    ):
        self.target_fps = target_fps
        self.enable_shadows = enable_shadows
        self.enable_lighting = enable_lighting
        self.opacity = opacity

        self._pose_engine = YoloPoseEngine(
            model_name=pose_model,
            device=device,
            conf_threshold=0.45,
            smooth_alpha=0.4,
        )
        self._async_pose = AsyncPoseEngine(self._pose_engine)
        self._densepose = DensePoseEngine(use_densepose=True)
        self._parsing = ParsingEngine(model_path=parsing_model, device=device)
        self._garment_analyzer = GarmentAnalyzer()
        self._warper = HybridWarper(smooth_alpha=0.35, physics_lag=0.2, tps_smooth=0.1)
        self._occlusion = OcclusionEngine(feather_radius=14)
        self._shadow = ShadowEngine(shadow_intensity=0.35)

        self._garments: List[GarmentEntry] = []
        self._current_idx: int = 0

        self._fps_counter = FPSCounter(window=30)
        self._frame_cache = FrameCache(change_threshold=4.0)
        self._last_pose: Optional[PoseKeypoints] = None
        self._last_parsing = None
        self._last_torso = None
        self._parse_frame_skip = 3
        self._frame_count = 0
        self._processing_scale = 1.0
        self._high_fps_mode = False
        self._debug_overlays: Dict[str, bool] = {
            "yolo": False,
            "parser": False,
            "densepose": False,
            "detectron": False,
        }

        self.stats = PipelineStats()
        self._models_loaded = False
        logger.info("RenderPipeline initialized")

    # ── Model Loading ────────────────────────────────────────────────────────

    def load_models(self) -> bool:
        logger.info("Loading AI models...")
        success = self._pose_engine.load()
        if not success:
            logger.error("YOLO pose model failed to load!")
            return False
        self._async_pose.start()
        self._models_loaded = True
        logger.info("Models loaded successfully")
        return True

    def unload_models(self):
        self._async_pose.stop()

    # ── Garment Catalog ──────────────────────────────────────────────────────

    def load_garments(self, shirts_dir: str) -> int:
        self._garments.clear()
        shirts_path = Path(shirts_dir)
        shirts_path.mkdir(parents=True, exist_ok=True)

        png_files = sorted(shirts_path.glob("*.png"))
        logger.info(f"Found {len(png_files)} shirt(s) in {shirts_dir}")
        for png_path in png_files:
            self._load_single_garment(str(png_path))

        if not self._garments:
            logger.info("No shirts found - generating placeholder shirts")
            self._generate_placeholder_shirts(shirts_path)

        if self._garments:
            self._preanalyze_garments()

        logger.info(f"Loaded {len(self._garments)} garment(s)")
        return len(self._garments)

    def _load_single_garment(self, path: str) -> bool:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Could not load: {path}")
                return False

            img = ensure_bgra(img)
            name = Path(path).stem

            # FIX: collar_y_ratio of ~0.08 means collar top is 8% from top of image.
            # These defaults match a typical shirt PNG where the neck opening is
            # near the very top.  Adjust per-garment via meta if needed.
            meta = GarmentMeta(
                path=path,
                name=name,
                collar_y_ratio=0.08,
                shoulder_y_ratio=0.18,
                sleeve_end_ratio=0.52,
                hem_y_ratio=0.96,
                neck_x_ratio=0.50,
            )

            thumb = cv2.resize(img, (80, 100))
            entry = GarmentEntry(path=path, name=name, image=img, meta=meta, thumbnail=thumb)
            self._garments.append(entry)
            logger.debug(f"Loaded shirt: {name}")
            return True
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return False

    def _generate_placeholder_shirts(self, shirts_path: Path):
        colors = [
            ((30, 80, 180), "Blue_Formal"),
            ((20, 120, 50), "Green_Casual"),
            ((180, 50, 30), "Red_Sport"),
            ((120, 30, 120), "Purple_Fashion"),
            ((30, 120, 150), "Teal_Business"),
        ]
        for color, name in colors:
            shirt_img = create_placeholder_shirt(size=(400, 500), color=color)
            save_path = shirts_path / f"{name}.png"
            cv2.imwrite(str(save_path), shirt_img)
            logger.info(f"Created placeholder: {name}.png")
            self._load_single_garment(str(save_path))

    def _preanalyze_garments(self):
        for entry in self._garments:
            try:
                landmarks = self._garment_analyzer.analyze(
                    entry.image, entry.meta, cache_key=entry.path,
                )
                entry.landmarks = landmarks
            except Exception as e:
                logger.error(f"Landmark analysis failed for {entry.name}: {e}")

    def add_garment(self, path: str) -> bool:
        result = self._load_single_garment(path)
        if result and self._garments:
            entry = self._garments[-1]
            try:
                entry.landmarks = self._garment_analyzer.analyze(
                    entry.image, entry.meta, cache_key=path
                )
            except Exception as e:
                logger.error(f"Failed to analyze new garment: {e}")
        return result

    # ── Shirt Navigation ─────────────────────────────────────────────────────

    @property
    def current_garment(self) -> Optional[GarmentEntry]:
        if not self._garments:
            return None
        return self._garments[self._current_idx]

    @property
    def garment_count(self) -> int:
        return len(self._garments)

    def next_shirt(self):
        if self._garments:
            self._current_idx = (self._current_idx + 1) % len(self._garments)
            self._warper.reset()
            logger.info(f"Shirt: {self.current_garment.name}")

    def previous_shirt(self):
        if self._garments:
            self._current_idx = (self._current_idx - 1) % len(self._garments)
            self._warper.reset()
            logger.info(f"Shirt: {self.current_garment.name}")

    def select_shirt(self, idx: int):
        if 0 <= idx < len(self._garments):
            self._current_idx = idx
            self._warper.reset()

    def get_shirt_names(self) -> List[str]:
        return [g.name for g in self._garments]

    # ── Main Processing ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, PipelineStats]:
        import traceback as _tb
        t_total = time.perf_counter()
        self._frame_count += 1
        result = frame.copy()

        try:
            scale = float(np.clip(self._processing_scale, 0.5, 1.0))
            work_frame = frame
            upsample = False
            if scale < 0.999:
                h, w = frame.shape[:2]
                ws = max(320, int(w * scale))
                hs = max(240, int(h * scale))
                work_frame = cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA)
                upsample = True

            processed, stats = self._process_frame_inner(work_frame, work_frame.copy(), t_total)

            if upsample and processed is not None:
                processed = cv2.resize(
                    processed,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            return processed, stats

        except Exception:
            logger.error(f"Pipeline error:\n{_tb.format_exc()}")
            self.stats.fps = self._fps_counter.tick()
            return result, self.stats

    def _process_frame_inner(
        self, frame: np.ndarray, result: np.ndarray, t_total: float
    ) -> Tuple[np.ndarray, PipelineStats]:
        if not self._models_loaded or not self._garments:
            self.stats.fps = self._fps_counter.tick()
            return self._no_detection_overlay(result), self.stats

        garment = self.current_garment
        if garment is None or garment.landmarks is None:
            self.stats.fps = self._fps_counter.tick()
            return result, self.stats

        # ── Pose Detection ────────────────────────────────────────────────────
        t_pose = time.perf_counter()
        self._async_pose.submit_frame(frame)
        pose = self._async_pose.get_latest_pose()
        self.stats.pose_ms = (time.perf_counter() - t_pose) * 1000
        self.stats.pose_detected = pose is not None and pose.is_usable()
        self._last_pose = pose

        if not pose or not pose.is_usable():
            self.stats.fps = self._fps_counter.tick()
            return self._no_detection_overlay(result), self.stats

        # ── Parsing (throttled) ───────────────────────────────────────────────
        h, w = frame.shape[:2]
        cache_invalid = (
            self._last_parsing is None
            or self._last_torso is None
            or getattr(self._last_parsing, "torso",
                       np.zeros((0, 0), dtype=np.uint8)).shape != (h, w)
            or getattr(self._last_torso, "torso_mask",
                       np.zeros((0, 0), dtype=np.uint8)).shape != (h, w)
        )
        if cache_invalid or self._frame_count % self._parse_frame_skip == 0:
            self._last_parsing = self._parsing.parse(frame, pose)
            self._last_torso = self._densepose.estimate(
                frame, pose, parsing_mask=self._last_parsing,
            )

        parsed = self._last_parsing
        torso_map = self._last_torso

        # ── Warp Shirt ────────────────────────────────────────────────────────
        t_warp = time.perf_counter()
        warp_result = self._warper.warp(
            garment.image,
            garment.landmarks,
            pose,
            frame.shape,           # (h, w, c) — warper uses [:2]
        )
        self.stats.warp_ms = (time.perf_counter() - t_warp) * 1000

        if warp_result is None:
            self.stats.fps = self._fps_counter.tick()
            return result, self.stats

        # ── Lighting Adaptation ───────────────────────────────────────────────
        t_render = time.perf_counter()
        warped_shirt = warp_result.warped_shirt

        if self.enable_lighting:
            warped_shirt = self._shadow.adapt_shirt_lighting(
                warped_shirt, frame,
                warp_result.placement_x,
                warp_result.placement_y,
            )

        # ── Occlusion Masks ───────────────────────────────────────────────────
        occlusion_masks = self._occlusion.build_occlusion_masks(
            frame, pose, parsed, torso_map
        )

        # ── Composite ─────────────────────────────────────────────────────────
        result = self._occlusion.composite(
            result,
            warped_shirt,
            warp_result.placement_x,
            warp_result.placement_y,
            occlusion_masks,
            opacity=self.opacity,
        )

        # ── Shadows ───────────────────────────────────────────────────────────
        if self.enable_shadows:
            shirt_region = occlusion_masks.get(
                "shirt_region", np.zeros(frame.shape[:2], dtype=np.uint8)
            )
            result = self._shadow.apply_shadows(
                result, shirt_region, pose, warped_shirt
            )

        self.stats.render_ms = (time.perf_counter() - t_render) * 1000
        self.stats.total_ms  = (time.perf_counter() - t_total)  * 1000
        self.stats.fps        = self._fps_counter.tick()
        self.stats.active_shirt  = garment.name
        self.stats.engine_method = parsed.method

        result = self._apply_debug_overlays(result, pose, parsed, torso_map)
        return result, self.stats

    # ── Debug / Helpers ──────────────────────────────────────────────────────

    def _overlay_mask(self, frame, mask, color, alpha=0.35):
        if mask is None:
            return frame
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 10
        if not np.any(mask_bool):
            return frame
        out = frame.copy().astype(np.float32)
        color_arr = np.array(color, dtype=np.float32)
        out[mask_bool] = out[mask_bool] * (1.0 - alpha) + color_arr * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_debug_overlays(self, frame, pose, parsed, torso_map):
        out = frame
        if self._debug_overlays.get("parser") and parsed is not None:
            out = self._overlay_mask(out, getattr(parsed, "torso", None),      (50, 170, 255), 0.28)
            out = self._overlay_mask(out, getattr(parsed, "left_arm", None),   (90, 230, 70),  0.32)
            out = self._overlay_mask(out, getattr(parsed, "right_arm", None),  (70, 220, 230), 0.32)
            out = self._overlay_mask(out, getattr(parsed, "face", None),       (40, 170, 255), 0.28)
            out = self._overlay_mask(out, getattr(parsed, "hair", None),       (170, 30, 170), 0.28)
            cv2.putText(out, "Parser", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 255), 2)

        if (self._debug_overlays.get("densepose") or
                self._debug_overlays.get("detectron")) and torso_map is not None:
            out = self._overlay_mask(out, getattr(torso_map, "torso_mask", None), (255, 120, 0),  0.24)
            arm_masks = getattr(torso_map, "arm_masks", {}) or {}
            out = self._overlay_mask(out, arm_masks.get("left_arm"),  (0, 255, 255), 0.24)
            out = self._overlay_mask(out, arm_masks.get("right_arm"), (0, 255, 0),   0.24)
            out = self._overlay_mask(out, getattr(torso_map, "neck_mask", None), (255, 0, 255), 0.24)
            label = "Detectron2 DensePose" if (
                self._debug_overlays.get("detectron") and self._densepose.has_densepose
            ) else "DensePose"
            cv2.putText(out, label, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 190, 20), 2)

        if self._debug_overlays.get("yolo") and pose is not None:
            out = self._pose_engine.draw_skeleton(out, pose)

        return out

    def _no_detection_overlay(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = frame.shape[:2]
        overlay = result.copy()
        cv2.rectangle(overlay, (w // 2 - 200, h - 60), (w // 2 + 200, h - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
        cv2.putText(result, "Stand in front of camera",
                    (w // 2 - 175, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return result

    # ── Public API ───────────────────────────────────────────────────────────

    def get_garment_thumbnails(self):
        return [g.thumbnail for g in self._garments]

    def get_model_status(self):
        return {
            "pose":   self._pose_engine.get_status(),
            "parsing": self._parsing.get_status(),
            "densepose": {"available": self._densepose.has_densepose},
            "garments_loaded": len(self._garments),
        }

    def toggle_shadows(self):   self.enable_shadows  = not self.enable_shadows
    def toggle_lighting(self):  self.enable_lighting = not self.enable_lighting
    def set_opacity(self, v):   self.opacity = float(np.clip(v, 0.1, 1.0))
    def set_processing_scale(self, s): self._processing_scale = float(np.clip(s, 0.5, 1.0))
    def set_parse_frame_skip(self, n): self._parse_frame_skip = int(np.clip(n, 1, 8))

    def set_high_fps_mode(self, enabled: bool):
        self._high_fps_mode = bool(enabled)
        if self._high_fps_mode:
            self.set_processing_scale(0.75)
            self.set_parse_frame_skip(5)
        else:
            self.set_processing_scale(1.0)
            self.set_parse_frame_skip(3)

    def set_debug_overlay(self, name: str, enabled: bool):
        key = str(name).strip().lower()
        if key in self._debug_overlays:
            self._debug_overlays[key] = bool(enabled)

    def get_debug_overlays(self):
        return dict(self._debug_overlays)

    def take_screenshot(self, frame: np.ndarray, output_dir: str = "screenshots") -> str:
        Path(output_dir).mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = str(Path(output_dir) / f"tryon_{ts}.png")
        cv2.imwrite(path, frame)
        logger.info(f"Screenshot saved: {path}")
        return path


__all__ = ["RenderPipeline", "PipelineStats", "GarmentEntry"]