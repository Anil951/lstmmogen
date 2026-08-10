"""
segment_image.py - Separate Image Segmentation Tool for Puppet Animation

This script isolates the image segmentation phase:
1. Loads a target character image and normalized keypoints.
2. Removes white background.
3. Computes pixel coordinate mapping.
4. Performs Watershed segmentation with interactive GUI refinement.
5. Stores the output segmentation data (cropped parts and joints) in a pickle cache
   so it can be used independently for any animation later.
"""

import os
import math
import cv2
import numpy as np
import pickle

from ActionConditionedLSTM.config import (
    BACKGROUND_THRESHOLD, DEFAULT_BRUSH_SIZE, MORPH_KERNEL_SIZE,
    WATERSHED_ERODE_SIZE, MASK_DILATE_SIZE, TARGET_IMAGE_PATH,
    NORMALIZED_KEYPOINTS_PATH, SKELETON_BONES, KEYPOINT_NAMES
)


# 1. Background Removal

def remove_white_background(img_path: str) -> np.ndarray:
    """
    Loads an image and converts a white background (RGB > 240) to an Alpha channel mask.
    """
    print(f"Loading image from {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {img_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lower_white = np.array([BACKGROUND_THRESHOLD, BACKGROUND_THRESHOLD, BACKGROUND_THRESHOLD], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(img_rgb, lower_white, upper_white)
    fg_mask = cv2.bitwise_not(white_mask)
    
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    return np.dstack([img_rgb, fg_mask])


# 2. Keypoint Mapping

def extract_source_keypoints(img_rgba: np.ndarray, json_path: str) -> np.ndarray:
    """
    Maps normalized keypoints from JSON onto actual pixel coordinates on the image canvas.
    """
    print(f"Reading keypoints from {json_path}...")
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    norm_kpts = np.array(data['keypoints'], dtype=np.float32)
    
    ys, xs = np.where(img_rgba[:, :, 3] > 0)
    if len(xs) == 0:
        raise ValueError("No foreground found in image to map keypoints to.")
        
    img_min_x, img_max_x = xs.min(), xs.max()
    img_min_y, img_max_y = ys.min(), ys.max()
    
    valid_kpts = np.array([pt for pt in norm_kpts if not (pt[0] == 0 and pt[1] == 0)])
    norm_min_x, norm_max_x = valid_kpts[:, 0].min(), valid_kpts[:, 0].max()
    norm_min_y, norm_max_y = valid_kpts[:, 1].min(), valid_kpts[:, 1].max()
    
    norm_h = norm_max_y - norm_min_y
    img_h = img_max_y - img_min_y
    
    S = img_h / norm_h if norm_h > 0 else 1.0
    
    norm_cx = (norm_max_x + norm_min_x) / 2.0
    norm_cy = (norm_max_y + norm_min_y) / 2.0
    
    img_cx = (img_max_x + img_min_x) / 2.0
    img_cy = (img_max_y + img_min_y) / 2.0
    
    pixel_kpts = []
    for nx, ny in norm_kpts:
        if nx == 0 and ny == 0:
            pixel_kpts.append([0.0, 0.0])
            continue
            
        px = (nx - norm_cx) * S + img_cx
        py = -(ny - norm_cy) * S + img_cy
        pixel_kpts.append([px, py])
        
    return np.array(pixel_kpts, dtype=np.float32)


# 3. Interactive Segmentation GUI

def refine_segmentation_gui(img_rgb: np.ndarray, markers: np.ndarray, part_ids: dict) -> np.ndarray:
    """
    Opens an interactive OpenCV window to refine Watershed body part segmentations.
    """
    window_name = "Refine Segmentation - Press ENTER to finish"
    id_to_name = {v: k for k, v in part_ids.items()}
    id_to_name[255] = "background"
    
    colors = {255: (0, 0, 0)}
    np.random.seed(42)
    for p_id in part_ids.values():
        colors[p_id] = tuple(np.random.randint(50, 255, 3).tolist())
        
    state = {
        'drawing': False,
        'current_label': 255,
        'brush_size': DEFAULT_BRUSH_SIZE,
        'markers': markers.copy()
    }
    
    def get_colored_overlay(m):
        overlay = np.zeros_like(img_rgb)
        for val, color in colors.items():
            overlay[m == val] = color
        overlay[m == -1] = (0, 0, 255)
        return overlay
        
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            cv2.circle(state['markers'], (x, y), state['brush_size'], int(state['current_label']), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if state['drawing']:
                cv2.circle(state['markers'], (x, y), state['brush_size'], int(state['current_label']), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            state['drawing'] = False
            cv2.circle(state['markers'], (x, y), state['brush_size'], int(state['current_label']), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            lbl = int(state['markers'][y, x])
            if lbl in id_to_name:
                state['current_label'] = lbl
                print(f"Selected label: {id_to_name[lbl]}")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n--- SEGMENTATION REFINEMENT MODE ---")
    print("Left-click & drag to paint. Right-click to pick a part label from the image.")
    print("Press '+' or '-' to change brush size.")
    print("Press ENTER to finish.")
    
    while True:
        overlay = get_colored_overlay(state['markers'])
        display = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)
        
        label_name = id_to_name.get(state['current_label'], "unknown")
        cv2.putText(display, f"Label: {label_name} | Size: {state['brush_size']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        
        if key == 13:
            break
        elif key in (ord('+'), ord('=')):
            state['brush_size'] += 2
        elif key in (ord('-'), ord('_')):
            state['brush_size'] = max(1, state['brush_size'] - 2)
            
    cv2.destroyAllWindows()
    return state['markers']


def segment_body_parts(img_rgba: np.ndarray, kpts: np.ndarray) -> dict:
    """
    Performs markers preparation, Watershed algorithm, GUI adjustment, and extracts cropped parts.
    """
    h, w = img_rgba.shape[:2]
    img_rgb = img_rgba[:, :, :3].copy()
    
    markers = np.zeros((h, w), dtype=np.int32)
    fg_mask = img_rgba[:, :, 3] > 128
    markers[~fg_mask] = 255
    
    part_ids = {}
    current_id = 1
    
    # Torso Marker
    torso_mask = np.zeros((h, w), dtype=np.uint8)
    corners = np.array([kpts[1], kpts[2], kpts[8], kpts[7]], dtype=np.int32)
    cv2.fillConvexPoly(torso_mask, corners, 255)
    torso_mask = cv2.erode(torso_mask, np.ones(WATERSHED_ERODE_SIZE if 'WATERSW_ERODE_SIZE' in locals() else WATERSHED_ERODE_SIZE, np.uint8), iterations=1)
    markers[torso_mask > 0] = current_id
    part_ids['torso'] = current_id
    current_id += 1
    
    # Face Marker
    face_pt = kpts[0]
    neck_pt = (kpts[1] + kpts[2]) / 2.0
    head_radius = max(int(np.linalg.norm(face_pt - neck_pt) * 0.7), 12)
    cv2.circle(markers, (int(face_pt[0]), int(face_pt[1])), head_radius, current_id, -1)
    part_ids['face'] = current_id
    current_id += 1

    # Limb Markers
    def draw_bone_marker(p1, p2, width, p_id, is_extremity=False):
        length = np.linalg.norm(p2 - p1)
        if length < 5: 
            return
        start = p1.copy()
        end = p2 + 0.35 * (p2 - p1) if is_extremity else p2.copy()
        cv2.line(markers, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), p_id, width)

    bones = {}
    bone_keys = ['L_upper_arm', 'L_lower_arm', 'R_upper_arm', 'R_lower_arm', 'L_upper_leg', 'L_lower_leg', 'R_upper_leg', 'R_lower_leg']
    # Rebuild bones dynamically mapping the key names to config SKELETON_BONES
    bone_indices = {
        'L_upper_arm': (1, 3), 'L_lower_arm': (3, 5),
        'R_upper_arm': (2, 4), 'R_lower_arm': (4, 6),
        'L_upper_leg': (7, 9), 'L_lower_leg': (9, 11),
        'R_upper_leg': (8, 10), 'R_lower_leg': (10, 12),
    }
    for b_name in bone_keys:
        if b_name in bone_indices:
            bones[b_name] = bone_indices[b_name]
    
    for name, (j1, j2) in bones.items():
        is_ext = j2 in [5, 6, 11, 12]
        w_val = 8 if is_ext else 5
        draw_bone_marker(kpts[j1], kpts[j2], width=w_val, p_id=current_id, is_extremity=is_ext)
        part_ids[name] = current_id
        current_id += 1

    # Run Watershed and let user refine
    cv2.watershed(img_rgb, markers)
    markers = refine_segmentation_gui(img_rgb, markers, part_ids)
    
    parts = {}
    for name, p_id in part_ids.items():
        part_mask = np.zeros((h, w), dtype=np.uint8)
        part_mask[markers == p_id] = 255
        
        ys, xs = np.where(part_mask > 0)
        if len(xs) == 0: 
            continue
        
        x1, x2 = max(0, xs.min()), min(w, xs.max() + 1)
        y1, y2 = max(0, ys.min()), min(h, ys.max() + 1)
        
        cropped = img_rgba[y1:y2, x1:x2].copy()
        
        dilated_mask = cv2.dilate(part_mask, np.ones(MASK_DILATE_SIZE, np.uint8), iterations=1)
        cropped[:, :, 3] = np.minimum(cropped[:, :, 3], dilated_mask[y1:y2, x1:x2])
        
        if name in bones:
            j1, j2 = bones[name]
            p1_crop = kpts[j1] - np.array([x1, y1])
            p2_crop = kpts[j2] - np.array([x1, y1])
            delta = kpts[j2] - kpts[j1]
            angle = math.degrees(math.atan2(delta[1], delta[0]))
            parts[name] = {'patch': cropped, 'p1': p1_crop, 'p2': p2_crop,
                           'ref_angle': angle, 'joints': (j1, j2)}
        elif name == 'torso':
            joints_in_crop = {
                'L_Sho': kpts[1] - np.array([x1, y1]),
                'R_Sho': kpts[2] - np.array([x1, y1]),
                'L_Hip': kpts[7] - np.array([x1, y1]),
                'R_Hip': kpts[8] - np.array([x1, y1])
            }
            delta = ((kpts[1]+kpts[2])/2.0) - ((kpts[7]+kpts[8])/2.0)
            ref_angle = math.degrees(math.atan2(delta[1], delta[0]))
            center = np.mean([joints_in_crop['L_Sho'], joints_in_crop['R_Sho'],
                              joints_in_crop['L_Hip'], joints_in_crop['R_Hip']], axis=0)
            parts['torso'] = {'patch': cropped, 'center': center,
                              'joints': joints_in_crop, 'ref_angle': ref_angle}
        elif name == 'face':
            parts['face'] = {'patch': cropped, 'center': kpts[0] - np.array([x1, y1])}
            
    # Save debug crops
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, 'iofiles', 'debug_parts')
    os.makedirs(debug_dir, exist_ok=True)
    for name, part in parts.items():
        cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), part['patch'])
            
    return parts


# 4. Main Segmentation Process

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    iofiles_dir = os.path.join(script_dir, 'iofiles')
    
    # 1. Paths configurations (Target image & normalized keypoints JSON)
    target_image = TARGET_IMAGE_PATH
    json_path = NORMALIZED_KEYPOINTS_PATH
    
    base_name = os.path.splitext(os.path.basename(target_image))[0]
    segmentation_cache = os.path.join(iofiles_dir, f"{base_name}_segmentation.pkl")
    
    if os.path.exists(segmentation_cache):
        print(f"Segmentation cache found at {segmentation_cache}. Using it directly.")
        return
        
    print(f"Segmenting target image: {target_image}")
    
    # 2. Preprocess image background
    img_rgba = remove_white_background(target_image)
    
    # 3. Map keypoints & segment body parts
    source_kpts = extract_source_keypoints(img_rgba, json_path)
    parts = segment_body_parts(img_rgba, source_kpts)
    
    # 4. Cache output
    print(f"Saving segmentation cache to {segmentation_cache}...")
    cache_data = {
        'parts': parts,
        'source_kpts': source_kpts
    }
    with open(segmentation_cache, 'wb') as f:
        pickle.dump(cache_data, f)
        
    print("Segmentation completed and cached successfully!")


if __name__ == "__main__":
    main()
