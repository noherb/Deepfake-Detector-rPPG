import os
import cv2
import numpy as np
import pickle
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from src.face_extractor import FaceExtractor
from src.signal_processor import SignalProcessor

def extract_windowed_features_from_video(video_path: str, model_path: str, window_sec: int = 5, step_sec: int = 2) -> List[List[float]]:
    """
    Extracts multiple 6-dimensional feature vectors from a video using sliding windows.
    Returns a list of feature vectors.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    extractor = FaceExtractor(model_path)
    processor = SignalProcessor(fps=fps)

    raw_signals = {"forehead": [], "left_cheek": [], "right_cheek": [], "chin": []}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rois = extractor.extract_rois_rgb(frame)
        if rois:
            for region, rgb in rois.items():
                raw_signals[region].append(rgb)

    cap.release()
    extractor.close()

    min_length = min(len(sig) for sig in raw_signals.values())
    window_frames = int(window_sec * fps)
    step_frames = int(step_sec * fps)

    if min_length < window_frames:
        return []

    filtered_signals = {}
    for region, rgb_list in raw_signals.items():
        rgb_array = np.array(rgb_list[:min_length])
        pos_sig = processor.apply_pos_algorithm(rgb_array)
        clean_sig = processor.butter_bandpass_filter(pos_sig)
        filtered_signals[region] = clean_sig

    window_features_list = []
    for i in range(0, min_length - window_frames + 1, step_frames):
        window_signals = {}
        for region, sig in filtered_signals.items():
            window_signals[region] = sig[i:i + window_frames]

        features = processor.compute_coherence_features(window_signals)
        window_features_list.append(features)

    return window_features_list


def train_fake_detector(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_output_path: str = "fake_detector_rf.pkl"):
    """
    Trains a Random Forest classifier using GroupKFold cross-validation.
    """
    gkf = GroupKFold(n_splits=5)
    accuracies = []
    best_model = None
    best_acc = 0.0

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # replaced SVM by random forest
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold + 1} Accuracy: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = clf

    print(f"\nMean Accuracy on folds: {np.mean(accuracies) * 100:.2f}%")

    os.makedirs(os.path.dirname(model_output_path) if os.path.dirname(model_output_path) else ".", exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved as '{model_output_path}'.")