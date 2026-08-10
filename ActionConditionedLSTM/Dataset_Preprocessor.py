import os
import glob
import numpy as np

from config import (
    ACTION_CLASSES,
    CLASS_MAP,
    SEQUENCE_LENGTH,
    CATEGORIZED_DATASET_DIR,
    PROCESSED_DATASET_PATH
)

def extract_13_keypoints_2d(motion_data_3d):
    """
    Extracts 13 2D keypoints (Front View: X, Y) from 24 3D HumanAct12 joints.
    13 Keypoint Order:
      0: Face (avg of Joint 12 Neck and Joint 15 Head)
      1: L_Shoulder (Joint 16)
      2: R_Shoulder (Joint 17)
      3: L_Elbow (Joint 18)
      4: R_Elbow (Joint 19)
      5: L_Wrist (Joint 20)
      6: R_Wrist (Joint 21)
      7: L_Hip (Joint 1)
      8: R_Hip (Joint 2)
      9: L_Knee (Joint 4)
      10: R_Knee (Joint 5)
      11: L_Ankle (Joint 7)
      12: R_Ankle (Joint 8)
    """
    T = motion_data_3d.shape[0]
    pose_13_3d = np.zeros((T, 13, 3), dtype=np.float32)
    
    pose_13_3d[:, 0, :] = (motion_data_3d[:, 12, :] + motion_data_3d[:, 15, :]) / 2.0
    pose_13_3d[:, 1, :] = motion_data_3d[:, 16, :]
    pose_13_3d[:, 2, :] = motion_data_3d[:, 17, :]
    pose_13_3d[:, 3, :] = motion_data_3d[:, 18, :]
    pose_13_3d[:, 4, :] = motion_data_3d[:, 19, :]
    pose_13_3d[:, 5, :] = motion_data_3d[:, 20, :]
    pose_13_3d[:, 6, :] = motion_data_3d[:, 21, :]
    pose_13_3d[:, 7, :] = motion_data_3d[:, 1, :]
    pose_13_3d[:, 8, :] = motion_data_3d[:, 2, :]
    pose_13_3d[:, 9, :] = motion_data_3d[:, 4, :]
    pose_13_3d[:, 10, :] = motion_data_3d[:, 5, :]
    pose_13_3d[:, 11, :] = motion_data_3d[:, 7, :]
    pose_13_3d[:, 12, :] = motion_data_3d[:, 8, :]

    # Front View 2D coordinates: X (axis 0) and Y (axis 1)
    kpts_2d = pose_13_3d[:, :, :2].copy()
    return kpts_2d

def normalize_keypoints_pose_extractor(pts_frame):
    """
    Exact normalization logic matching pose_extractor.py:
    1. Shift keypoints so Hip midpoint (indices 7 & 8) is at (0, 0).
    2. Scale keypoints by torso length (norm of Shoulder mid to Hip mid).
    3. Flip Y-axis so positive Y is UP (Head) and negative Y is DOWN (Feet).
    """
    pts = pts_frame.copy()
    hip_mid = (pts[7] + pts[8]) / 2.0
    shoulder_mid = (pts[1] + pts[2]) / 2.0

    pts = pts - hip_mid
    shoulder_mid_shifted = shoulder_mid - hip_mid
    torso_length = np.linalg.norm(shoulder_mid_shifted)

    if torso_length < 1e-6:
        torso_length = 1.0

    pts = pts / torso_length
    pts[:, 1] = -pts[:, 1]  # Flip Y so Head is UP
    return pts

def create_action_sequences(dataset_dir, seq_length=SEQUENCE_LENGTH):
    X_list = []
    y_list = []
    actions_list = []

    total_files = 0
    for cls_name, cls_idx in CLASS_MAP.items():
        cls_folder = os.path.join(dataset_dir, cls_name)
        files = sorted(glob.glob(os.path.join(cls_folder, '*.npy')))
        print(f"Processing category '{cls_name}' ({len(files)} files)...")

        one_hot = np.zeros(len(ACTION_CLASSES), dtype=np.float32)
        one_hot[cls_idx] = 1.0

        for file_path in files:
            total_files += 1
            motion = np.load(file_path)
            if motion.shape[0] <= seq_length + 1:
                continue

            kpts_2d = extract_13_keypoints_2d(motion)
            norm_kpts = np.array([normalize_keypoints_pose_extractor(f) for f in kpts_2d], dtype=np.float32)

            # Create overlapping sequences of length seq_length
            for i in range(len(norm_kpts) - seq_length - 1):
                X_seq = norm_kpts[i : i + seq_length]
                y_seq = norm_kpts[i + 1 : i + seq_length + 1]
                action_seq = np.tile(one_hot, (seq_length, 1))

                X_list.append(X_seq)
                y_list.append(y_seq)
                actions_list.append(action_seq)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)

    print(f"\nPreprocessing Complete from {total_files} motion clips!")
    print(f"X shape: {X.shape} (Sequences, Seq_Len, Joints, 2)")
    print(f"y shape: {y.shape}")
    print(f"Actions shape: {actions.shape} (Sequences, Seq_Len, Num_Classes)")
    print(f"X value range: min={X.min():.4f}, max={X.max():.4f}")
    print(f"Sample Frame 0 Face Y (should be >0): {X[0, 0, 0, 1]:.4f}, L_Ankle Y (should be <0): {X[0, 0, 11, 1]:.4f}")

    return X, y, actions

def main():
    X, y, actions = create_action_sequences(CATEGORIZED_DATASET_DIR)
    np.savez(PROCESSED_DATASET_PATH, X=X, y=y, actions=actions)
    print(f"Saved processed dataset to '{PROCESSED_DATASET_PATH}'")

if __name__ == "__main__":
    main()
