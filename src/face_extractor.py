import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceExtractor:
    """
    Handles face detection and multi-Region of Interest (ROI) extraction using MediaPipe.
    Extracts the average RGB color of specific facial patches (forehead, cheeks, chin).
    """

    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.roi_landmarks = {
            "forehead": 151,
            "left_cheek": 118,
            "right_cheek": 347,
            "chin": 152
        }

    def _get_roi_mean_rgb(self, frame: np.ndarray, landmarks, landmark_idx: int, box_size: int = 20) -> Optional[
        Tuple[float, float, float]]:
        """Helper to extract the mean RGB value of a specific bounding box around a landmark."""
        h, w, _ = frame.shape
        point_x = int(landmarks[landmark_idx].x * w)
        point_y = int(landmarks[landmark_idx].y * h)

        x1, y1 = max(0, point_x - box_size), max(0, point_y - box_size)
        x2, y2 = min(w, point_x + box_size), min(h, point_y + box_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])

        return (mean_r, mean_g, mean_b)

    def extract_rois_rgb(self, frame: np.ndarray) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """
        Detects the face and returns the mean RGB values for multiple ROIs.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            Dictionary containing the RGB tuples for each ROI (e.g., {'forehead': (R, G, B), ...}).
            Returns None if the face is lost or if any ROI fails to extract.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]

        extracted_rois = {}
        for region_name, landmark_idx in self.roi_landmarks.items():
            rgb_vals = self._get_roi_mean_rgb(frame, landmarks, landmark_idx, box_size=20)
            if rgb_vals:
                extracted_rois[region_name] = rgb_vals

        if len(extracted_rois) == len(self.roi_landmarks):
            return extracted_rois

        return None

    def close(self):
        """Releases the MediaPipe resources."""
        self.detector.close()
