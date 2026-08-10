import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ActionConditionedRNN.losses import weighted_loss

ACTION_CLASSES = ['run', 'walk', 'jump_vertical']
CLASS_MAP = {cls_name: idx for idx, cls_name in enumerate(ACTION_CLASSES)}

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
    def __init__(self, model_path, sequence_length=20, num_joints=13, num_classes=3):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    iofiles_dir = os.path.join(workspace_dir, 'iofiles')
    os.makedirs(iofiles_dir, exist_ok=True)

    model_path = os.path.join(iofiles_dir, 'action_rnn_model.h5')
    output_npy = os.path.join(iofiles_dir, 'generated_action_motion.npy')
    seed_info_path = os.path.join(iofiles_dir, 'seed_info.npz')

    dataset_file = os.path.join(iofiles_dir, 'processed_dataset.npz')
    if not os.path.exists(dataset_file):
        print(f"Error: Processed dataset '{dataset_file}' not found. Please run Dataset_Preprocessor.py first.")
        return

    data = np.load(dataset_file)
    X_all = data['X']         # (N, 20, 13, 2)
    y_all = data['y']         # (N, 20, 13, 2)
    actions_all = data['actions'] # (N, 20, 3)

    # Action choice: 'run', 'walk', or 'jump_vertical'
    action_choice = 'walk'
    cls_idx = CLASS_MAP[action_choice]

    # Filter samples belonging to the chosen action class
    action_mask = actions_all[:, 0, cls_idx] == 1.0
    X_class = X_all[action_mask]
    y_class = y_all[action_mask]

    # Randomly pick a sample index from this class
    rand_idx = np.random.randint(0, len(X_class))
    seed_sequence = X_class[rand_idx]  # (20, 13, 2)
    ground_truth_next = y_class[rand_idx]  # (20, 13, 2) — the real next-frame targets

    print(f"Action: '{action_choice}', Class samples: {len(X_class)}, Random seed index: {rand_idx}")
    print(f"Seed sequence shape: {seed_sequence.shape}")

    predictor = ActionPosePredictor(model_path=model_path)

    num_gen_frames = 100
    print(f"\nGenerating {num_gen_frames} frames of animation for action: '{action_choice}'...")

    generated = predictor.generate_points(seed_sequence, action_name=action_choice, num_iterations=num_gen_frames)

    # Save generated motion
    np.save(output_npy, generated)
    print(f"Saved generated motion (shape: {generated.shape}) to '{output_npy}'")

    # Save seed info for side-by-side preview comparison
    np.savez(seed_info_path,
             seed_sequence=seed_sequence,
             ground_truth_next=ground_truth_next,
             generated=generated,
             action_name=np.array(action_choice))
    print(f"Saved seed info for preview comparison to '{seed_info_path}'")

if __name__ == "__main__":
    main()
