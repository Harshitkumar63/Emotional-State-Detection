"""
Face Preprocessor — Face Detection & Cropping
===============================================
Uses MTCNN (Multi-task Cascaded Convolutional Networks) to detect and
crop faces from images before emotion analysis.

Why face detection matters:
  Emotion models are trained on tightly-cropped face images (FER-2013).
  If we feed a full scene photo (person sitting at desk), the model sees
  background pixels that confuse classification.  MTCNN detects the face
  bounding box so we can crop to just the face region.

Graceful fallback:
  If MTCNN is not installed or no face is detected, the preprocessor
  returns the original image with a warning.  The system still works —
  just with potentially reduced accuracy.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from PIL import Image
import numpy as np

from src.utils.helpers import setup_logging

logger = setup_logging()


class FacePreprocessor:
    """Detect and crop faces from images for emotion analysis."""

    def __init__(self, margin: int = 40, min_face_size: int = 40):
        """
        Parameters
        ----------
        margin : int
            Pixels to add around the detected face bounding box.
        min_face_size : int
            Minimum face size (in pixels) to consider a valid detection.
        """
        self.margin = margin
        self.min_face_size = min_face_size
        self._mtcnn = None
        self._mtcnn_available = False

        self._try_load_mtcnn()

    def _try_load_mtcnn(self):
        """Attempt to load MTCNN; fail silently if unavailable."""
        try:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                keep_all=False,       # return only the most prominent face
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
            )
            self._mtcnn_available = True
            logger.info("MTCNN face detector loaded.")
        except ImportError:
            logger.info(
                "facenet-pytorch not installed. Face detection disabled. "
                "Install with: pip install facenet-pytorch"
            )
        except Exception as e:
            logger.warning("Could not initialise MTCNN: %s", e)

    def process(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Detect and crop the face from the image.

        Parameters
        ----------
        image_input : str or PIL.Image
            File path or PIL Image.

        Returns
        -------
        PIL.Image : cropped face region (or original if no face detected).
        """
        image = self._load_image(image_input)

        if not self._mtcnn_available:
            logger.info("MTCNN not available; returning original image.")
            return self._resize(image)

        try:
            boxes, probs = self._mtcnn.detect(image)

            if boxes is None or len(boxes) == 0:
                logger.warning("No face detected; returning original image.")
                return self._resize(image)

            # Use the highest-confidence detection
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            confidence = probs[best_idx]

            logger.info("Face detected (confidence: %.2f)", confidence)

            # Crop with margin
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = image.size
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(w, x2 + self.margin)
            y2 = min(h, y2 + self.margin)

            face = image.crop((x1, y1, x2, y2))
            return self._resize(face)

        except Exception as e:
            logger.warning("Face detection failed (%s); returning original image.", e)
            return self._resize(image)

    def validate(self, image_input: Union[str, Image.Image]) -> dict:
        """Check if a face can be detected in the image.

        Returns
        -------
        dict with keys: face_detected (bool), confidence, bbox, issues
        """
        image = self._load_image(image_input)
        issues = []

        if not self._mtcnn_available:
            return {
                "face_detected": None,
                "confidence": 0.0,
                "bbox": None,
                "issues": ["MTCNN not available; cannot validate face detection"],
            }

        try:
            boxes, probs = self._mtcnn.detect(image)

            if boxes is None or len(boxes) == 0:
                return {
                    "face_detected": False,
                    "confidence": 0.0,
                    "bbox": None,
                    "issues": ["No face detected in image"],
                }

            best_idx = np.argmax(probs)
            conf = float(probs[best_idx])
            bbox = [int(v) for v in boxes[best_idx]]

            if conf < 0.7:
                issues.append("Low face detection confidence")

            # Check face size relative to image
            face_w = bbox[2] - bbox[0]
            face_h = bbox[3] - bbox[1]
            img_w, img_h = image.size
            face_ratio = (face_w * face_h) / (img_w * img_h)

            if face_ratio < 0.05:
                issues.append("Face is very small relative to image")

            return {
                "face_detected": True,
                "confidence": round(conf, 3),
                "bbox": bbox,
                "issues": issues,
            }

        except Exception as e:
            return {
                "face_detected": None,
                "confidence": 0.0,
                "bbox": None,
                "issues": [f"Face detection error: {e}"],
            }

    @property
    def is_available(self) -> bool:
        """Whether MTCNN face detection is available."""
        return self._mtcnn_available

    @staticmethod
    def _load_image(image_input: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Expected file path or PIL Image, got {type(image_input)}")

    @staticmethod
    def _resize(image: Image.Image, target_size: int = 224) -> Image.Image:
        """Resize to the model's expected input size while preserving aspect ratio."""
        return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
