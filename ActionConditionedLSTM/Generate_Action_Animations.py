import os
import glob
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from losses import weighted_loss
from dataset_preprocessor import extract_13_keypoints_2d, normalize_keypoints_pose_extractor
from config import (
    ACTION_CLASSES,
    CLASS_MAP,
    SEQUENCE_LENGTH,
    NUM_JOINTS,
    NUM_CLASSES,
    MODEL_PATH,
    GENERATED_MOTION_PATH,
    SEED_INFO_PATH,
    WORKSPACE_DIR,
    ACTION_CHOICE
)

def normalize_keypoints_pose_extractor(pts_frame):
    """
    Exact normalization matching pose_extractor.py:
    1. Shift keypoints so Hip midpoint (indices 7 & 8) is at (0, 0).
    2. Scale keypoints by torso length.
    3. Flip Y-axis so positive Y is UP (Head) and negative Y is DOWN (Feet).
    """
    pts = np.array(pts_frame, dtype=np.float32).copy()
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

class ActionPosePredictor:
    def __init__(self, model_path, sequence_length=SEQUENCE_LENGTH, num_joints=NUM_JOINTS, num_classes=NUM_CLASSES):
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.model_path = model_path
        self.model = self.load_action_model()

    def load_action_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading Action RNN model from '{self.model_path}'...")
            return load_model(self.model_path, custom_objects={'weighted_loss': weighted_loss})
        else:
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'. Please run Train_Action_Model.py first.")

    def generate_points(self, seed_sequence, action_name, num_iterations=100):
        """
        Generates num_iterations pose frames conditioned on a 20-frame seed sequence and action_name.
        seed_sequence: shape (20, 13, 2) — a real 20-frame sequence from the dataset
        action_name: 'run', 'walk', or 'jump_vertical'
        Returns: generated_points of shape (num_iterations, 13, 2)
        """
        if action_name not in CLASS_MAP:
            raise ValueError(f"Unknown action '{action_name}'. Valid actions are: {ACTION_CLASSES}")

        cls_idx = CLASS_MAP[action_name]
        one_hot = np.zeros((1, self.sequence_length, self.num_classes), dtype=np.float32)
        one_hot[0, :, cls_idx] = 1.0

        # Use the full 20-frame seed sequence directly (already normalized from dataset)
        input_pose_seq = seed_sequence.reshape(1, self.sequence_length, self.num_joints, 2).astype(np.float32)

        generated_points = np.zeros((num_iterations, self.num_joints, 2), dtype=np.float32)

        for i in range(num_iterations):
            out_seq = self.model.predict([input_pose_seq, one_hot], verbose=0)
            pred_pose = out_seq[0, -1, :, :]
            
            # --- FIX: ENFORCE SCALE AND CENTER ---
            # Models naturally regress to the mean (shrink) when predicting under uncertainty.
            # We must re-center and re-scale the prediction so the model always receives a perfectly sized skeleton for the next step.
            hip_mid = (pred_pose[7] + pred_pose[8]) / 2.0
            pred_pose = pred_pose - hip_mid
            torso_length = np.linalg.norm((pred_pose[1] + pred_pose[2]) / 2.0)
            if torso_length > 1e-6:
                pred_pose = pred_pose / torso_length
            
            generated_points[i] = pred_pose

            # Shift sliding window forward by 1 frame
            input_pose_seq = np.concatenate(
                [input_pose_seq[:, 1:, :, :], pred_pose.reshape(1, 1, self.num_joints, 2)],
                axis=1
            )

        return generated_points

def load_initial_frame_from_json(json_path):
    """Loads keypoints extracted by pose_extractor.py saved as normalized_keypoints.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    kpts = np.array(data['keypoints'], dtype=np.float32) # (13, 2)
    return kpts

def main():
    action_choice = ACTION_CHOICE
    cls_idx = CLASS_MAP[action_choice]

    raw_dataset_dir = os.path.join(WORKSPACE_DIR, 'dataset', 'HumanAct12_Categorized', action_choice)
    files = glob.glob(os.path.join(raw_dataset_dir, '*.npy'))
    if not files:
        print(f"Error: No raw dataset files found for action '{action_choice}' in {raw_dataset_dir}")
        return

    # Find files that are long enough to provide a meaningful ground truth continuation
    num_gen_frames = 60
    valid_files = [f for f in files if np.load(f).shape[0] >= SEQUENCE_LENGTH + 10]
    if not valid_files:
        valid_files = files  # fallback if none are long enough

    rand_file = valid_files[np.random.randint(0, len(valid_files))]
    motion_3d = np.load(rand_file)
    print(f"Action: '{action_choice}', Selected raw file: {os.path.basename(rand_file)} (Total frames: {motion_3d.shape[0]})")

    # Extract and normalize directly
    kpts_2d = extract_13_keypoints_2d(motion_3d)
    norm_kpts = np.array([normalize_keypoints_pose_extractor(f) for f in kpts_2d], dtype=np.float32)

    # Split into seed (first SEQUENCE_LENGTH frames) and true ground truth future (remaining frames)
    seed_sequence = norm_kpts[:SEQUENCE_LENGTH]
    ground_truth_next = norm_kpts[SEQUENCE_LENGTH : SEQUENCE_LENGTH + num_gen_frames]

    print(f"Seed sequence shape: {seed_sequence.shape}")
    print(f"True Ground truth future shape: {ground_truth_next.shape}")

    predictor = ActionPosePredictor(model_path=MODEL_PATH)

    print(f"\nGenerating {num_gen_frames} frames of animation for action: '{action_choice}'...")

    generated = predictor.generate_points(seed_sequence, action_name=action_choice, num_iterations=num_gen_frames)

    # Save generated motion
    output_npy = GENERATED_MOTION_PATH
    np.save(output_npy, generated)
    print(f"Saved generated motion (shape: {generated.shape}) to '{output_npy}'")

    # Save seed info for side-by-side preview comparison
    np.savez(SEED_INFO_PATH,
             seed_sequence=seed_sequence,
             ground_truth_next=ground_truth_next,
             generated=generated,
             action_name=np.array(action_choice))
    print(f"Saved seed info for preview comparison to '{SEED_INFO_PATH}'")

if __name__ == "__main__":
    main()
