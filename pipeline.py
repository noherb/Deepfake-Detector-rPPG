import os
import cv2
import numpy as np
from scipy import signal
from typing import List, Tuple, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceExtractor:
    """
    Handles face detection and Region of Interest (ROI) extraction using MediaPipe.
    Extracts the average color of a specific facial patch (e.g., forehead).
    """

    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_forehead_green(self, frame: np.ndarray) -> Optional[float]:
        """
        Detects the face in the frame and returns the mean green channel value of the forehead ROI.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            Mean value of the green channel in the ROI, or None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape

        forehead_x = int(landmarks[151].x * w)
        forehead_y = int(landmarks[151].y * h)

        box_size = 30
        x1, y1 = max(0, forehead_x - box_size), max(0, forehead_y - box_size)
        x2, y2 = min(w, forehead_x + box_size), min(h, forehead_y + box_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return np.mean(roi[:, :, 1])

    def close(self):
        """Releases the MediaPipe resources."""
        self.detector.close()


class SignalProcessor:
    """
    Processes the raw optical signal to extract the heart rate (BPM).
    Applies detrending, bandpass filtering, and Fourier transform.
    """

    def __init__(self, fps: float, min_hz: float = 0.7, max_hz: float = 3.0):
        self.fps = fps
        self.min_hz = min_hz
        self.max_hz = max_hz

    def estimate_bpm(self, raw_signal: List[float]) -> Tuple[float, np.ndarray]:
        """
        Calculates the Beats Per Minute (BPM) from a raw color signal.

        Args:
            raw_signal: List of mean color values per frame.

        Returns:
            Tuple containing the estimated BPM and the filtered signal array.
        """
        signal_array = np.array(raw_signal)

        detrended = signal.detrend(signal_array)

        b, a = signal.butter(4, [self.min_hz, self.max_hz], btype='bandpass', fs=self.fps)
        filtered = signal.filtfilt(b, a, detrended)

        fft_values = np.fft.rfft(filtered)
        frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / self.fps)

        fft_values[0] = 0

        peak_idx = np.argmax(np.abs(fft_values))
        peak_freq = frequencies[peak_idx]

        bpm = peak_freq * 60.0
        return bpm, filtered

    def estimate_bpm_windowed(self, raw_signal: List[float], window_sec: int = 10, step_sec: int = 2) -> float:
        """
        Calculates the BPM using sliding windows to prevent global noise (like motion artifacts)
        from overwhelming the cardiac signal.

        Args:
            raw_signal: List of mean color values per frame.
            window_sec: Size of the sliding window in seconds.
            step_sec: Step size between consecutive windows in seconds.

        Returns:
            The estimated heart rate (BPM) computed as the median of all windowed estimates.
        """
        window_frames = int(window_sec * self.fps)
        step_frames = int(step_sec * self.fps)

        if len(raw_signal) < window_frames:
            bpm, _ = self.estimate_bpm(raw_signal)
            return bpm

        bpms = []
        for i in range(0, len(raw_signal) - window_frames + 1, step_frames):
            window_signal = raw_signal[i:i + window_frames]
            bpm, _ = self.estimate_bpm(window_signal)
            bpms.append(bpm)

        return float(np.median(bpms))


def read_ubfc_ground_truth(txt_path: str) -> float:
    """
    Reads the ground truth file from the UBFC-rPPG dataset and returns the average true BPM.

    Args:
        txt_path: Path to the ground_truth.txt file.

    Returns:
        Average true Heart Rate (BPM) over the session.
    """
    hr_values = []

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) >= 2:
                hr_values.append(float(parts[1]))

    if not hr_values:
        raise ValueError("The ground truth file is empty or bad formatted.")

    return np.mean(hr_values)


def evaluate_video(video_path: str, gt_path: str, model_path: str):
    """
    Runs the complete pipeline on a single video and computes the Mean Absolute Error (MAE).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    extractor = FaceExtractor(model_path)
    processor = SignalProcessor(fps=fps)

    raw_signal = []

    print(f"Extracting signal from {os.path.basename(video_path)}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        green_val = extractor.extract_forehead_green(frame)
        if green_val is not None:
            raw_signal.append(green_val)
        elif len(raw_signal) > 0:
            # in case we lose signal
            raw_signal.append(raw_signal[-1])

    cap.release()
    extractor.close()

    if len(raw_signal) == 0:
        print("Error: No face detected.")
        return

    predicted_bpm = processor.estimate_bpm_windowed(raw_signal, window_sec=10, step_sec=2)

    true_bpm = read_ubfc_ground_truth(gt_path)

    mae = abs(predicted_bpm - true_bpm)

    print("-" * 30)
    print(f"Real mean BPM: {true_bpm:.1f}")
    print(f"Estimated BPM: {predicted_bpm:.1f}")
    print(f"Absolute Error (MAE): {mae:.1f} BPM")
    print("-" * 30)


if __name__ == "__main__":
    MODEL_PATH = "face_landmarker.task"

    VIDEO_FILE = "assets/5-gt/vid-001.avi"
    GT_FILE = "assets/5-gt/gtdump.xmp"

    if os.path.exists(VIDEO_FILE) and os.path.exists(GT_FILE):
        evaluate_video(VIDEO_FILE, GT_FILE, MODEL_PATH)
    else:
        print(f"Please update VIDEO_FILE and GT_FILE paths.")