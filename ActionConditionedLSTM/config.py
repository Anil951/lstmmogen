import os

# =============================================================================
# Base Directories
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
IOFILES_DIR = os.path.join(WORKSPACE_DIR, 'iofiles')
DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset', 'HumanAct12')
CATEGORIZED_DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset', 'HumanAct12_Categorized')

# =============================================================================
# YOLO Pose Estimation Configurations
# =============================================================================
YOLO_WEIGHTS_DIR = IOFILES_DIR
YOLO_DETECTION_MODEL = 'yolo11x.pt'
YOLO_POSE_MODEL = 'yolo11x-pose.pt'
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = 'cpu'

# =============================================================================
# Character Image Configurations & Default Paths
# =============================================================================
TARGET_IMAGE_NAME = '006027f5-13d6-43e8-a3b8-223f377428b7_resized.jpg'
TARGET_IMAGE_PATH = os.path.join(IOFILES_DIR, TARGET_IMAGE_NAME)
CROPPED_IMAGE_PATH = os.path.join(IOFILES_DIR, f'cropped_{TARGET_IMAGE_NAME}')

# =============================================================================
# File Paths
# =============================================================================
PROCESSED_DATASET_PATH = os.path.join(IOFILES_DIR, 'processed_dataset.npz')
MODEL_PATH = os.path.join(IOFILES_DIR, 'action_rnn_model.h5')
SEED_INFO_PATH = os.path.join(IOFILES_DIR, 'seed_info.npz')
GENERATED_MOTION_PATH = os.path.join(IOFILES_DIR, 'jump_motion.npy')
TRAINING_HISTORY_PATH = os.path.join(IOFILES_DIR, 'training_history.json')
TRAINING_METRICS_PATH = os.path.join(IOFILES_DIR, 'training_metrics.png')
COMPARISON_GIF_PATH = os.path.join(IOFILES_DIR, 'comparison.gif')
NORMALIZED_KEYPOINTS_PATH = os.path.join(IOFILES_DIR, 'normalized_keypoints.json')
ANIMATED_CHARACTER_PATH = os.path.join(IOFILES_DIR, 'B_jump.gif')

# =============================================================================
# Thresholds & Parameters
# =============================================================================
BACKGROUND_THRESHOLD = 240
DEFAULT_BRUSH_SIZE = 5
MORPH_KERNEL_SIZE = (5, 5)
WATERSHED_ERODE_SIZE = (9, 9)
MASK_DILATE_SIZE = (5, 5)
TORSO_EXPANSION_FACTOR = 1.03
LIMB_OVERLAP_FACTOR = 1.05
MANUAL_ANNOTATION_PADDING = 200

# =============================================================================
# Model & Data Parameters
# =============================================================================
SEQUENCE_LENGTH = 20
NUM_JOINTS = 13
POSE_DIM = NUM_JOINTS * 2
TARGET_SAMPLING_PROB = 0.5

# =============================================================================
# Actions Config
# =============================================================================
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

# =============================================================================
# Keypoints Topology
# =============================================================================
KEYPOINT_NAMES = [
    "Face", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", 
    "L_Ankle", "R_Ankle"
]

SKELETON_BONES = [
    (0, 1), (0, 2), (1, 2),          # Face to shoulders & shoulder line
    (1, 3), (3, 5), (2, 4), (4, 6),  # Arms
    (1, 7), (2, 8), (7, 8),          # Torso
    (7, 9), (9, 11), (8, 10), (10, 12) # Legs
]

# =============================================================================
# GIF Composer Configurations
# =============================================================================
COMPOSER_GIF_SEQUENCE = ['A_walk.gif', 'B_boxing.gif', 'A_run.gif', 'B_jump.gif']
COMPOSER_OUTPUT_PATH = os.path.join(IOFILES_DIR, 'story_combined.gif')
COMPOSER_FPS = 15

# Flag to enable/disable skeleton overlay (bones and joints) on final animation
OVERLAP_SKELETON = False

# Dynamic action choice for motion generation
ACTION_CHOICE = 'jump_vertical'
