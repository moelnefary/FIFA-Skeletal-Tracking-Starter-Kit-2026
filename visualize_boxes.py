"""
Visualize bounding boxes overlaid on video frames.
Supports both:
  1. Keypoints NPZ files (computes boxes from keypoints + draws skeleton)
  2. Pre-computed boxes .npy files (draws boxes only)

Usage:
    # From keypoints NPZ (with skeleton):
    python visualize_boxes.py \
        --keypoints_file data/keypoints/ARG_CRO_220001_15keypoints_FULL.npz \
        --image_dir data/images/ARG_CRO_220001 \
        --output_dir data/viz_boxes

    # From pre-computed boxes .npy (boxes only):
    python visualize_boxes.py \
        --boxes_file data/keypoints/ENG_FRA_223257.npy \
        --image_dir data/images/ENG_FRA_223257 \
        --output_dir data/viz_boxes
"""

import argparse
from pathlib import Path
import numpy as np
import cv2


# 15 keypoint connections for skeleton drawing
SKELETON_CONNECTIONS = [
    (0, 1),   # Nose -> RShoulder
    (0, 2),   # Nose -> LShoulder
    (1, 3),   # RShoulder -> RElbow
    (2, 4),   # LShoulder -> LElbow
    (3, 5),   # RElbow -> RWrist
    (4, 6),   # LElbow -> LWrist
    (1, 7),   # RShoulder -> RHip
    (2, 8),   # LShoulder -> LHip
    (7, 8),   # RHip -> LHip
    (7, 9),   # RHip -> RKnee
    (8, 10),  # LHip -> LKnee
    (9, 11),  # RKnee -> RAnkle
    (10, 12), # LKnee -> LAnkle
    (11, 13), # RAnkle -> RBigToe
    (12, 14), # LAnkle -> LBigToe
]

# Colors for different players (BGR)
PLAYER_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 255, 0),  # Light green
    (255, 128, 0),  # Light blue
    (0, 128, 255),  # Orange
    (128, 0, 255),  # Purple
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 255),
    (200, 200, 0),
    (200, 0, 200),
    (0, 200, 200),
    (100, 255, 100),
    (255, 100, 100),
    (100, 100, 255),
    (180, 180, 100),
    (180, 100, 180),
    (100, 180, 180),
]


def extract_boxes_from_keypoints(j2d, padding=20, image_w=1920, image_h=1080):
    """Compute bounding boxes from 2D keypoints. j2d shape: (persons, frames, K, 2)"""
    with np.errstate(all="ignore"):
        x_min = np.nanmin(j2d[..., 0], axis=-1) - padding
        y_min = np.nanmin(j2d[..., 1], axis=-1) - padding
        x_max = np.nanmax(j2d[..., 0], axis=-1) + padding
        y_max = np.nanmax(j2d[..., 1], axis=-1) + padding

    x_min = np.clip(x_min, 0, image_w)
    y_min = np.clip(y_min, 0, image_h)
    x_max = np.clip(x_max, 0, image_w)
    y_max = np.clip(y_max, 0, image_h)

    boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)
    # Transpose from (persons, frames, 4) to (frames, persons, 4)
    boxes = boxes.transpose(1, 0, 2)

    nan_mask = np.all(np.isnan(j2d), axis=(-1, -2))  # (persons, frames)
    nan_mask = nan_mask.T  # (frames, persons)
    boxes[nan_mask] = np.nan

    return boxes


def draw_frame(image, boxes_frame, keypoints_frame, frame_idx):
    """
    Draw bounding boxes and optionally skeleton keypoints on a single frame.

    Args:
        image: (H, W, 3) BGR image
        boxes_frame: (NUM_PERSONS, 4) — [x1, y1, x2, y2]
        keypoints_frame: (NUM_PERSONS, K, 2) — 2D keypoints, or None for boxes-only mode
        frame_idx: int
    """
    vis = image.copy()
    num_persons = boxes_frame.shape[0]

    for p in range(num_persons):
        box = boxes_frame[p]

        # Skip if all NaN or all zero
        if np.isnan(box).any() or np.all(box == 0):
            continue

        color = PLAYER_COLORS[p % len(PLAYER_COLORS)]
        x1, y1, x2, y2 = box.astype(int)

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw player ID
        cv2.putText(vis, f"P{p}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, color, 1, cv2.LINE_AA)

        # Draw skeleton keypoints (if available)
        if keypoints_frame is not None:
            kps = keypoints_frame[p]
            if not np.all(np.isnan(kps)):
                for ki in range(kps.shape[0]):
                    if not np.isnan(kps[ki, 0]):
                        cx, cy = int(kps[ki, 0]), int(kps[ki, 1])
                        cv2.circle(vis, (cx, cy), 3, color, -1)

                # Draw skeleton connections
                for (a, b) in SKELETON_CONNECTIONS:
                    if a < kps.shape[0] and b < kps.shape[0]:
                        if not np.isnan(kps[a, 0]) and not np.isnan(kps[b, 0]):
                            pt1 = (int(kps[a, 0]), int(kps[a, 1]))
                            pt2 = (int(kps[b, 0]), int(kps[b, 1]))
                            cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)

    # Add frame number
    cv2.putText(vis, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes on video frames")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keypoints_file", type=str,
                       help="Path to *_15keypoints_FULL.npz file (computes boxes + draws skeleton)")
    group.add_argument("--boxes_file", type=str,
                       help="Path to pre-computed boxes .npy file (draws boxes only)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing extracted frames (jpg)")
    parser.add_argument("--output_dir", type=str, default="data/viz_boxes",
                        help="Output directory for visualization images")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sample frames to visualize")
    parser.add_argument("--padding", type=int, default=20,
                        help="Padding around skeleton bounding box (pixels)")
    parser.add_argument("--specific_frames", type=str, default=None,
                        help="Comma-separated specific frame indices to visualize")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    j2d_t = None  # Will be set if using keypoints mode

    if args.keypoints_file:
        # Mode 1: From keypoints NPZ — compute boxes + draw skeleton
        keypoints_path = Path(args.keypoints_file)
        data = np.load(keypoints_path, allow_pickle=True)
        j2d = data["joints_2d_pixels"]  # (persons, frames, 15, 2)
        num_persons, num_frames = j2d.shape[0], j2d.shape[1]
        print(f"Keypoints shape: {j2d.shape} => {num_persons} persons, {num_frames} frames")

        boxes = extract_boxes_from_keypoints(j2d, padding=args.padding)
        j2d_t = j2d.transpose(1, 0, 2, 3)  # (frames, persons, 15, 2)
        print(f"Boxes shape: {boxes.shape}")
    else:
        # Mode 2: From pre-computed boxes .npy — draw boxes only
        boxes_path = Path(args.boxes_file)
        boxes = np.load(boxes_path)  # (frames, persons, 4)
        num_frames, num_persons = boxes.shape[0], boxes.shape[1]
        print(f"Boxes shape: {boxes.shape} => {num_frames} frames, {num_persons} persons")

    # Get image files
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"Found {len(image_files)} image files")

    if len(image_files) < num_frames:
        print(f"WARNING: Only {len(image_files)} images but {num_frames} frames in data!")
        num_frames = min(num_frames, len(image_files))

    # Select frames to visualize
    if args.specific_frames:
        frame_indices = [int(x) for x in args.specific_frames.split(",")]
    else:
        frame_indices = np.linspace(0, num_frames - 1, args.num_samples, dtype=int).tolist()

    print(f"Visualizing frames: {frame_indices}")
    print("-" * 60)

    for frame_idx in frame_indices:
        img_path = image_files[frame_idx]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Frame {frame_idx}: could not read {img_path}")
            continue

        kps_frame = j2d_t[frame_idx] if j2d_t is not None else None
        vis = draw_frame(image, boxes[frame_idx], kps_frame, frame_idx)

        # Count visible players
        valid = ~(np.isnan(boxes[frame_idx]).any(axis=-1) | np.all(boxes[frame_idx] == 0, axis=-1))
        visible = valid.sum()

        out_path = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Frame {frame_idx}: {visible} visible players -> {out_path}")

    print("-" * 60)
    print(f"Done! {len(frame_indices)} visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
