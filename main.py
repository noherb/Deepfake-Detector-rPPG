import os
import numpy as np
from src.pipeline import extract_windowed_features_from_video, train_fake_detector

if __name__ == "__main__":
    ORIGINAL_DIR = "assets/original/kaggle/input/datasets/xdxd003/ff-c23/faceforensics++_c23/original/"
    DEEPFAKE_DIR = "assets/deepfakes/kaggle/input/datasets/xdxd003/ff-c23/faceforensics++_c23/deepfakes/"
    MODEL_PATH = "face_landmarker.task"

    X = []
    y = []
    groups = []

    print("Beginning feature extraction...")

    if os.path.exists(ORIGINAL_DIR):
        original_files = sorted(os.listdir(ORIGINAL_DIR))
        processed_couples = 0

        for idx, filename in enumerate(original_files):
            if not filename.endswith(('.mp4', '.avi', '.mov')):
                continue

            orig_path = os.path.join(ORIGINAL_DIR, filename)
            video_id = os.path.splitext(filename)[0]

            window_feats_orig = extract_windowed_features_from_video(orig_path, MODEL_PATH)

            if window_feats_orig:
                matching_deepfakes = [f for f in os.listdir(DEEPFAKE_DIR) if
                                      f == f"{video_id}.mp4" or f.startswith(f"{video_id}_")]
                df_path = os.path.join(DEEPFAKE_DIR, matching_deepfakes[0]) if matching_deepfakes else ""

                if os.path.exists(df_path):
                    window_feats_df = extract_windowed_features_from_video(df_path, MODEL_PATH)

                    if window_feats_df:
                        print(f"[{processed_couples + 1}] Treated pair: {filename} ({len(window_feats_orig)} real windows, {len(window_feats_df)} fake windows)")

                        for feats in window_feats_orig:
                            X.append(feats)
                            y.append(0)
                            groups.append(video_id)

                        for feats in window_feats_df:
                            X.append(feats)
                            y.append(1)
                            groups.append(video_id)

                        processed_couples += 1

            if processed_couples >= 10:
                break
    else:
        print(f"Error: {ORIGINAL_DIR} is not a directory.")

    X_arr = np.array(X)
    y_arr = np.array(y)
    groups_arr = np.array(groups)

    print(f"\nEnd of extraction. Total number of samples (windows) generated: {len(X_arr)}")

    if len(X_arr) > 10:
        print("Launch of training with Random Forest and GroupKFold...")
        train_fake_detector(X_arr, y_arr, groups_arr, model_output_path="models/fake_detector_rf.pkl")
    else:
        print("Not enough samples extracted to start training. Check your paths.")