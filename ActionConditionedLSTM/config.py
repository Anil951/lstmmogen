import os

# Base Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
IOFILES_DIR = os.path.join(WORKSPACE_DIR, 'iofiles')
DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset', 'HumanAct12')
CATEGORIZED_DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset', 'HumanAct12_Categorized')

# File Paths
PROCESSED_DATASET_PATH = os.path.join(IOFILES_DIR, 'processed_dataset.npz')
MODEL_PATH = os.path.join(IOFILES_DIR, 'action_rnn_model.h5')
SEED_INFO_PATH = os.path.join(IOFILES_DIR, 'seed_info.npz')
GENERATED_MOTION_PATH = os.path.join(IOFILES_DIR, 'generated_action_motion.npy')
TRAINING_HISTORY_PATH = os.path.join(IOFILES_DIR, 'training_history.json')
TRAINING_METRICS_PATH = os.path.join(IOFILES_DIR, 'training_metrics.png')
COMPARISON_GIF_PATH = os.path.join(IOFILES_DIR, 'comparison.gif')

# Model & Data Parameters
SEQUENCE_LENGTH = 20
NUM_JOINTS = 13
POSE_DIM = NUM_JOINTS * 2
TARGET_SAMPLING_PROB = 0.5

# Actions Config
TARGET_ACTIONS = {
    "0201": "walk",
    "0301": "run",
    "0402": "jump_vertical",
    "1202": "throw_both_hands",
    "1104": "boxing_right_left",
    "0401": "jump_handsup",
}

ACTION_CLASSES = [
    'run', 'walk', 'jump_vertical',
    'throw_both_hands', 'boxing_right_left', 'jump_handsup'
]
NUM_CLASSES = len(ACTION_CLASSES)
CLASS_MAP = {cls_name: idx for idx, cls_name in enumerate(ACTION_CLASSES)}

# Kinematic structure (Skeleton topology)
SKELETON_BONES = [
    (0, 1), (0, 2), (1, 2),          # Face to shoulders & shoulder line
    (1, 3), (3, 5), (2, 4), (4, 6),  # Arms
    (1, 7), (2, 8), (7, 8),          # Torso
    (7, 9), (9, 11), (8, 10), (10, 12) # Legs
]
