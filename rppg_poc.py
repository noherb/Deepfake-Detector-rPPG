import os
import urllib.request
import time
from collections import deque
import cv2
import numpy as np
from scipy import signal

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class RPPGProcessor:
    def __init__(self, fps=30, window_seconds=10):
        self.fps = fps
        self.buffer_size = fps * window_seconds
        self.signal_buffer = deque(maxlen=self.buffer_size)

    def add_sample(self, value):
        self.signal_buffer.append(value)

    def estimate_heart_rate(self):
        if len(self.signal_buffer) < self.buffer_size:
            return None

        raw_signal = np.array(self.signal_buffer)
        detrended = signal.detrend(raw_signal)

        b, a = signal.butter(4, [0.7, 3.0], btype='bandpass', fs=self.fps)
        filtered = signal.filtfilt(b, a, detrended)

        fft_values = np.fft.rfft(filtered)
        frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / self.fps)

        fft_values[0] = 0

        peak_idx = np.argmax(np.abs(fft_values))
        peak_freq = frequencies[peak_idx]

        return peak_freq * 60


def download_model():
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading the MediaPipe Face Landmarker model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def main():
    model_path = download_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    processor = RPPGProcessor(fps=int(actual_fps), window_seconds=10)
    print(f"Starting at ~{actual_fps} FPS. Stay still...")

    start_time_ms = int(time.time() * 1000)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        current_time_ms = int(time.time() * 1000) - start_time_ms

        detection_result = detector.detect_for_video(mp_image, current_time_ms)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            h, w, _ = frame.shape

            forehead_x = int(landmarks[151].x * w)
            forehead_y = int(landmarks[151].y * h)

            box_size = 10
            x1 = max(0, forehead_x - box_size)
            y1 = max(0, forehead_y - box_size)
            x2 = min(w, forehead_x + box_size)
            y2 = min(h, forehead_y + box_size)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                green_avg = np.mean(roi[:, :, 1])
                processor.add_sample(green_avg)

                bpm = processor.estimate_heart_rate()

                if bpm is not None:
                    cv2.putText(frame, f"BPM: {bpm:.1f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    progress = len(processor.signal_buffer) / processor.buffer_size * 100
                    cv2.putText(frame, f"Calibrating... {int(progress)}%", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('rPPG PoC', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()