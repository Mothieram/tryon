"""
yolo_pose.py - YOLOv8 Pose Detection Engine
Handles body keypoint detection with CUDA support,
temporal smoothing, and efficient inference pipeline.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, List, Tuple, Dict,Any
from pathlib import Path
import threading

from engine.coreutils import (
    setup_logger,
    Keypoint,
    PoseKeypoints,
    smooth_array,
    FPSCounter,
)

logger = setup_logger("yolo_pose")


class YoloPoseEngine:
    """
    YOLOv8 Pose Detection Engine.

    Detects 17 COCO body keypoints per person using YOLOv8-pose model.
    Supports CUDA acceleration, confidence filtering, and temporal smoothing
    for stable real-time virtual try-on applications.
    """

    # COCO 17-keypoint names for reference
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        device: str = "auto",
        conf_threshold: float = 0.5,
        keypoint_conf: float = 0.3,
        smooth_alpha: float = 0.4,
        target_person_idx: int = 0,
    ):
        """
        Initialize pose engine.

        Args:
            model_name: YOLOv8 pose model filename (will auto-download)
            device: 'auto', 'cuda', 'cpu', or '0' for GPU index
            conf_threshold: Person detection confidence threshold
            keypoint_conf: Minimum keypoint confidence
            smooth_alpha: Temporal smoothing factor (0=max smooth, 1=no smooth)
            target_person_idx: Index of person to track (0=largest/most confident)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.keypoint_conf = keypoint_conf
        self.smooth_alpha = smooth_alpha
        self.target_person_idx = target_person_idx

        self._model = None
        self._device = self._resolve_device(device)
        self._is_loaded = False
        self._load_error: Optional[str] = None

        # Temporal smoothing state
        self._prev_keypoints: Optional[np.ndarray] = None
        self._smooth_weight = smooth_alpha

        # Performance tracking
        self.inference_ms: float = 0.0
        self._fps = FPSCounter(window=20)

        logger.info(f"YoloPoseEngine initialized | device={self._device} | model={model_name}")

    def _resolve_device(self, device: str) -> str:
        """Auto-detect best available device."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                return "cuda"
        except ImportError:
            pass
        logger.info("Using CPU inference")
        return "cpu"

    def load(self) -> bool:
        """Load YOLO model. Returns True on success."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading {self.model_name}...")
            self._model = YOLO(self.model_name)

            # Warm up model
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._model(
                dummy,
                device=self._device,
                verbose=False,
                conf=self.conf_threshold,
            )

            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self._device}")
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def detect(self, frame: np.ndarray) -> Optional[PoseKeypoints]:
        """
        Run pose detection on a single frame.

        Args:
            frame: BGR image array

        Returns:
            PoseKeypoints if person detected, else None
        """
        if not self._is_loaded or self._model is None:
            return None

        t0 = time.perf_counter()

        try:
            results = self._model(
                frame,
                device=self._device,
                verbose=False,
                conf=self.conf_threshold,
            )

            self.inference_ms = (time.perf_counter() - t0) * 1000
            self._fps.tick()

            pose = self._parse_results(results, frame.shape)
            return pose

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def _parse_results(
        self,
        results,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[PoseKeypoints]:
        """Parse YOLO results into PoseKeypoints."""
        if not results or len(results) == 0:
            return None

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Select best person (highest bounding box confidence)
        person_idx = self._select_best_person(result)
        if person_idx < 0:
            return None

        kp_data = result.keypoints.data[person_idx].cpu().numpy()  # (17, 3): x, y, conf
        box_conf = float(result.boxes.conf[person_idx].cpu().numpy()) if result.boxes is not None else 0.0

        # Build keypoints list
        keypoints = []
        raw_xy = np.zeros((17, 2), dtype=np.float32)

        for i, (x, y, conf) in enumerate(kp_data):
            kp = Keypoint(x=float(x), y=float(y), confidence=float(conf))
            keypoints.append(kp)
            raw_xy[i] = [x, y]

        # Temporal smoothing
        smoothed_xy = self._temporal_smooth(raw_xy)

        # Apply smoothed coordinates back
        smoothed_keypoints = []
        for i, kp in enumerate(keypoints):
            smoothed_kp = Keypoint(
                x=float(smoothed_xy[i, 0]),
                y=float(smoothed_xy[i, 1]),
                confidence=kp.confidence,
            )
            smoothed_keypoints.append(smoothed_kp)

        return PoseKeypoints(keypoints=smoothed_keypoints, confidence=box_conf)

    def _select_best_person(self, result) -> int:
        """Select the most prominent person in frame."""
        if result.boxes is None or len(result.boxes) == 0:
            return -1

        if len(result.boxes) == 1:
            conf = float(result.boxes.conf[0])
            return 0 if conf >= self.conf_threshold else -1

        # Select largest bounding box (closest person)
        areas = []
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = box.cpu().numpy()
            areas.append((x2 - x1) * (y2 - y1))

        best_idx = int(np.argmax(areas))
        conf = float(result.boxes.conf[best_idx])
        return best_idx if conf >= self.conf_threshold else -1

    def _temporal_smooth(self, raw_xy: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to keypoint positions."""
        if self._prev_keypoints is None:
            self._prev_keypoints = raw_xy.copy()
            return raw_xy

        # Only smooth visible keypoints (conf > 0 indicated by non-zero xy)
        visible = np.any(raw_xy > 0, axis=1)
        smoothed = self._prev_keypoints.copy()

        for i in range(17):
            if visible[i]:
                smoothed[i] = smooth_array(
                    self._prev_keypoints[i],
                    raw_xy[i],
                    alpha=self._smooth_weight,
                )

        self._prev_keypoints = smoothed
        return smoothed

    def reset_smoothing(self):
        """Reset temporal smoothing state (call on shirt change or reset)."""
        self._prev_keypoints = None

    def draw_skeleton(self, frame: np.ndarray, pose: PoseKeypoints) -> np.ndarray:
        """Draw skeleton visualization on frame for debugging."""
        vis = frame.copy()
        if not pose:
            return vis

        # Draw connections
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Arms
            (5, 11), (6, 12), (11, 12),                   # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),       # Legs
            (0, 5), (0, 6),                               # Neck
        ]

        for i, j in connections:
            kp1 = pose.keypoints[i] if i < len(pose.keypoints) else None
            kp2 = pose.keypoints[j] if j < len(pose.keypoints) else None
            if kp1 and kp2 and kp1.valid and kp2.valid:
                cv2.line(vis, kp1.to_tuple(), kp2.to_tuple(), (0, 255, 0), 2)

        # Draw keypoints
        colors = {
            (0, 4): (255, 100, 100),    # Head
            (5, 10): (100, 255, 100),   # Arms
            (11, 16): (100, 100, 255),  # Legs
        }
        for kp in pose.keypoints:
            if kp.valid:
                cv2.circle(vis, kp.to_tuple(), 4, (0, 255, 255), -1)

        # FPS overlay
        cv2.putText(
            vis,
            f"Pose FPS: {self._fps.fps:.0f} | Infer: {self.inference_ms:.0f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return vis

    @property
    def current_fps(self) -> float:
        return self._fps.fps

    def get_status(self) -> Dict[str, Any]:
        """Return engine status dict for UI."""
        return {
            "loaded": self._is_loaded,
            "device": self._device,
            "model": self.model_name,
            "fps": self.current_fps,
            "inference_ms": self.inference_ms,
            "error": self._load_error,
        }


# ─────────────────────────────────────────────
# Async Wrapper for non-blocking inference
# ─────────────────────────────────────────────

class AsyncPoseEngine:
    """
    Async wrapper around YoloPoseEngine.
    Runs inference in a background thread, returns last result.
    Prevents UI thread from blocking on model inference.
    """

    def __init__(self, engine: YoloPoseEngine):
        self.engine = engine
        self._last_pose: Optional[PoseKeypoints] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame

    def get_latest_pose(self) -> Optional[PoseKeypoints]:
        with self._lock:
            return self._last_pose

    def _inference_loop(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None:
                pose = self.engine.detect(frame)
                with self._lock:
                    self._last_pose = pose

            time.sleep(0.01)  # ~100Hz polling cap


__all__ = ["YoloPoseEngine", "AsyncPoseEngine"]