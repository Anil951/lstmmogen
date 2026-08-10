import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import (
    SKELETON_BONES,
    SEED_INFO_PATH,
    GENERATED_MOTION_PATH,
    COMPARISON_GIF_PATH
)

def preview_side_by_side(seed_info_path, save_gif=False, output_gif=None):
    """
    Side-by-side animation: Left = Ground Truth (dataset), Right = Generated.
    The seed 20-frame sequence plays first, then the continuation diverges:
      - Left panel continues with real dataset frames (ground_truth_next)
      - Right panel continues with model-generated frames
    """
    if not os.path.exists(seed_info_path):
        print(f"Error: Seed info file '{seed_info_path}' not found. Run Generate_Action_Animations.py first.")
        return

    info = np.load(seed_info_path, allow_pickle=True)
    seed_sequence = info['seed_sequence']      # (20, 13, 2)
    ground_truth_next = info['ground_truth_next']  # (20, 13, 2) — real next-frame targets
    generated = info['generated']              # (num_gen_frames, 13, 2)
    action_name = str(info['action_name'])

    # Build full sequences for comparison:
    gt_full = np.concatenate([seed_sequence, ground_truth_next], axis=0)

    # Generated: seed (20 frames) + generated frames
    gen_full = np.concatenate([seed_sequence, generated], axis=0)

    # Trim to same length for synchronized playback
    num_frames = min(len(gt_full), len(gen_full))
    gt_full = gt_full[:num_frames]
    gen_full = gen_full[:num_frames]
    seed_len = len(seed_sequence)

    # Compute axis limits from both sequences
    all_data = np.concatenate([gt_full, gen_full], axis=0)
    x_min, x_max = all_data[:, :, 0].min() - 0.3, all_data[:, :, 0].max() + 0.3
    y_min, y_max = all_data[:, :, 1].min() - 0.3, all_data[:, :, 1].max() + 0.3

    fig, (ax_gt, ax_gen) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Action: '{action_name}' — Ground Truth vs Generated", fontsize=14, fontweight='bold')

    for ax, title in [(ax_gt, "Ground Truth (Dataset)"), (ax_gen, "Generated (Model)")]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Create initial artists for both panels
    gt_bones = [ax_gt.plot([], [], 'g-', lw=2)[0] for _ in SKELETON_BONES]
    gt_scatter, = ax_gt.plot([], [], 'bo', ms=6)
    gen_bones = [ax_gen.plot([], [], 'r-', lw=2)[0] for _ in SKELETON_BONES]
    gen_scatter, = ax_gen.plot([], [], 'bo', ms=6)

    frame_text_gt = ax_gt.text(0.02, 0.98, '', transform=ax_gt.transAxes, fontsize=10, verticalalignment='top')
    frame_text_gen = ax_gen.text(0.02, 0.98, '', transform=ax_gen.transAxes, fontsize=10, verticalalignment='top')

    def update(frame):
        # Ground truth panel
        gt_pts = gt_full[frame]
        gt_scatter.set_data(gt_pts[:, 0], gt_pts[:, 1])
        for i, (start, end) in enumerate(SKELETON_BONES):
            gt_bones[i].set_data([gt_pts[start][0], gt_pts[end][0]], [gt_pts[start][1], gt_pts[end][1]])

        # Generated panel
        gen_pts = gen_full[frame]
        gen_scatter.set_data(gen_pts[:, 0], gen_pts[:, 1])
        for i, (start, end) in enumerate(SKELETON_BONES):
            gen_bones[i].set_data([gen_pts[start][0], gen_pts[end][0]], [gen_pts[start][1], gen_pts[end][1]])

        # Frame counter and phase label
        if frame < seed_len:
            phase = f"SEED (shared) frame {frame + 1}/{seed_len}"
        else:
            phase = f"CONTINUATION frame {frame - seed_len + 1}/{num_frames - seed_len}"

        frame_text_gt.set_text(phase)
        frame_text_gen.set_text(phase)

        return gt_bones + [gt_scatter] + gen_bones + [gen_scatter] + [frame_text_gt, frame_text_gen]

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=83.33, blit=False, repeat=True
    )

    if save_gif and output_gif:
        try:
            ani.save(output_gif, writer='pillow', fps=12)
            print(f"Comparison animation saved as GIF to '{output_gif}'")
        except Exception as e:
            print(f"Could not save GIF: {e}")

    try:
        plt.show()
    except Exception as e:
        print(f"Plot display window skipped ({e}).")


def preview_single(npy_path, save_gif=False, output_gif=None):
    """Single-panel animation preview."""
    if not os.path.exists(npy_path):
        print(f"Error: File '{npy_path}' not found.")
        return

    all_points = np.load(npy_path)
    print(f"Loaded animation sequence from '{npy_path}' with shape {all_points.shape}")

    fig, ax = plt.subplots(figsize=(6, 6))
    x_min, x_max = all_points[:, :, 0].min() - 0.3, all_points[:, :, 0].max() + 0.3
    y_min, y_max = all_points[:, :, 1].min() - 0.3, all_points[:, :, 1].max() + 0.3
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title("Action-Conditioned Stickman Animation", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    scatter, = ax.plot([], [], 'bo', ms=6)
    line_objects = [ax.plot([], [], 'r-', lw=2)[0] for _ in SKELETON_BONES]

    def update(frame):
        current_points = all_points[frame]
        scatter.set_data(current_points[:, 0], current_points[:, 1])
        for i, (start, end) in enumerate(SKELETON_BONES):
            line_objects[i].set_data(
                [current_points[start][0], current_points[end][0]],
                [current_points[start][1], current_points[end][1]]
            )
        return [scatter] + line_objects

    ani = animation.FuncAnimation(
        fig, update, frames=len(all_points), interval=33.33, blit=False
    )

    if save_gif and output_gif:
        try:
            ani.save(output_gif, writer='pillow', fps=30)
            print(f"Animation saved as GIF to '{output_gif}'")
        except Exception as e:
            print(f"Could not save GIF: {e}")

    try:
        plt.show()
    except Exception as e:
        print(f"Plot display window skipped ({e}).")


def main():
    if os.path.exists(SEED_INFO_PATH):
        preview_side_by_side(SEED_INFO_PATH, save_gif=True, output_gif=COMPARISON_GIF_PATH)
    else:
        gif_path = GENERATED_MOTION_PATH.replace('.npy', '.gif')
        preview_single(GENERATED_MOTION_PATH, save_gif=True, output_gif=gif_path)

if __name__ == "__main__":
    main()
