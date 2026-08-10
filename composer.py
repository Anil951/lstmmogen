"""
composer.py - GIF Sequencing and Stitching Tool

This script takes a sequence of generated GIF animations and stitches them 
together frame-by-frame in a specified order to produce a single combined GIF.

Configurations (e.g., input GIFs list, output path, frame rates) are loaded 
from config.py, with optional CLI overrides.
"""

import os
import argparse
import imageio
import cv2

# Import configurations
from ActionConditionedLSTM.config import (
    IOFILES_DIR, COMPOSER_GIF_SEQUENCE, COMPOSER_OUTPUT_PATH, COMPOSER_FPS
)


def compose_gifs(gif_sequence: list, output_path: str, fps: int = 15):
    """
    Reads a list of GIF file paths, extracts their frames, aligns their dimensions, 
    and stitches them sequentially into a single output GIF.
    """
    combined_frames = []
    target_width, target_height = None, None

    print(f"\nStarting GIF composition sequence: {gif_sequence}")

    for idx, gif_name in enumerate(gif_sequence):
        # Resolve full path (allows both absolute paths and relative names in iofiles)
        if os.path.isabs(gif_name):
            gif_path = gif_name
        else:
            gif_path = os.path.join(IOFILES_DIR, gif_name)

        if not os.path.exists(gif_path):
            print(f"Warning: GIF file '{gif_path}' not found. Skipping.")
            continue

        print(f"Reading: {gif_path}...")
        reader = imageio.get_reader(gif_path)
        frames = []
        for frame in reader:
            frames.append(frame)
        reader.close()

        if not frames:
            continue

        # Set target dimensions based on the first successfully loaded GIF
        if target_width is None or target_height is None:
            target_height, target_width = frames[0].shape[:2]
            print(f"Output dimensions set to: {target_width}x{target_height}")

        # Stitch and align frames
        for f_idx, frame in enumerate(frames):
            fh, fw = frame.shape[:2]
            if fw != target_width or fh != target_height:
                # Resize frame to match target resolution
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            combined_frames.append(frame)

    if not combined_frames:
        print("Error: No frames were successfully loaded. Output GIF was not generated.")
        return

    print(f"Stitching complete! Total frames: {len(combined_frames)}")
    print(f"Saving combined GIF to: {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, combined_frames, fps=fps, loop=0)
    print("Success!")

if __name__ == "__main__":
    compose_gifs(COMPOSER_GIF_SEQUENCE, COMPOSER_OUTPUT_PATH, COMPOSER_FPS)
