from ActionConditionedLSTM.config import GENERATED_MOTION_PATH
import numpy as np
import cv2
import os

def render_perspective_video(motion_data, bones, h_axis, v_axis, output_path, flip_h=False, flip_v=False):
    """
    Renders a 2D video from 3D data based on the chosen horizontal and vertical axes.
    h_axis / v_axis indices: 0 = X, 1 = Y, 2 = Z
    """
    frame_width, frame_height = 800, 800
    fps = 12  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Extract the specific axes we want for this view
    h_data = motion_data[:, :, h_axis]
    v_data = motion_data[:, :, v_axis]

    min_h, max_h = np.min(h_data), np.max(h_data)
    min_v, max_v = np.min(v_data), np.max(v_data)

    # Calculate uniform scaling factor to maintain aspect ratio
    scale_h = (frame_width - 100) / (max_h - min_h + 1e-5)
    scale_v = (frame_height - 100) / (max_v - min_v + 1e-5)
    scale = min(scale_h, scale_v)

    for frame_idx in range(motion_data.shape[0]):
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        current_joints = motion_data[frame_idx]
        pixel_joints = []

        for joint in current_joints:
            h_val = joint[h_axis]
            v_val = joint[v_axis]
            
            # Map to horizontal pixel coordinate
            px = (h_val - min_h) * scale + 50
            if flip_h:
                px = frame_width - px
            
            # Map to vertical pixel coordinate
            py = (v_val - min_v) * scale + 50
            if flip_v:
                py = frame_height - py

            pixel_joints.append((int(px), int(py)))

        # Draw bones
        for bone in bones:
            pt1 = pixel_joints[bone[0]]
            pt2 = pixel_joints[bone[1]]
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 3, lineType=cv2.LINE_AA)

        # Draw joints
        for pt in pixel_joints:
            cv2.circle(canvas, pt, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        out.write(canvas)

    out.release()
    print(f"Saved perspective video to: {output_path}")

def generate_multi_pov_videos(npy_filepath):
    try:
        motion_data = np.load(npy_filepath)
        print(f"Loaded data with shape: {motion_data.shape}")
    except Exception as e:
        print(f"Error loading {npy_filepath}: {e}")
        return

    # Convert 24 joints to 13 keypoints (matching pose_extractor & the project)
    pose_13 = np.zeros((motion_data.shape[0], 13, 3), dtype=np.float32)
    # 0 Face = avg(12, 15)
    pose_13[:, 0, :] = (motion_data[:, 12, :] + motion_data[:, 15, :]) / 2.0
    pose_13[:, 1, :] = motion_data[:, 16, :]  # L_Shoulder
    pose_13[:, 2, :] = motion_data[:, 17, :]  # R_Shoulder
    pose_13[:, 3, :] = motion_data[:, 18, :]  # L_Elbow
    pose_13[:, 4, :] = motion_data[:, 19, :]  # R_Elbow
    pose_13[:, 5, :] = motion_data[:, 20, :]  # L_Wrist (Fist)
    pose_13[:, 6, :] = motion_data[:, 21, :]  # R_Wrist (Fist)
    pose_13[:, 7, :] = motion_data[:, 1, :]   # L_Hip
    pose_13[:, 8, :] = motion_data[:, 2, :]   # R_Hip
    pose_13[:, 9, :] = motion_data[:, 4, :]   # L_Knee
    pose_13[:, 10, :] = motion_data[:, 5, :]  # R_Knee
    pose_13[:, 11, :] = motion_data[:, 7, :]  # L_Ankle
    pose_13[:, 12, :] = motion_data[:, 8, :]  # R_Ankle

    motion_data = pose_13

    # Define kinematic chains for 13 joints
    bones = [
        (0, 1), (0, 2),          
        (1, 3), (3, 5),          
        (2, 4), (4, 6),          
        (1, 7), (7, 9), (9, 11), 
        (2, 8), (8, 10), (10, 12),
        (7, 8)  # Optional: connect hips for better visualization
    ]

    # ==========================================
    # GENERATE THE FRONT VIEW
    # ==========================================
    
    # Front View (X vs Y)
    # Axes: X=0, Y=1
    render_perspective_video(
        motion_data, bones, 
        h_axis=0, v_axis=1, 
        output_path="front_view.mp4", 
        flip_h=False, flip_v=False
    )

if __name__ == "__main__":
    motion_file = GENERATED_MOTION_PATH
    if os.path.exists(motion_file):
        generate_multi_pov_videos(motion_file)
