"""
Colab-compatible pipeline script to prepare data for SAM-3D Body (preprocess.py).

This script:
  1. Extracts bounding boxes from keypoints NPZ files → data/boxes/<seq>.npy
  2. Extracts video frames from MP4 files → data/images/<seq>/*.jpg
  3. Generates data/sequences_full.txt listing all sequences

After running this, the data/ directory will be ready for preprocess.py (SAM).

Expected input (keypoints and videos can be in the same or separate folders):
    keypoints_dir/
    ├── ARG_CRO_220001_15keypoints_FULL.npz
    ├── MOR_POR_193202_15keypoints_FULL.npz
    └── ...

    videos_dir/
    ├── ARG_CRO_220001.mp4
    ├── MOR_POR_193202.mp4
    └── ...

Output structure (compatible with preprocess.py / SAM):
    output_dir/
    ├── sequences_full.txt
    ├── boxes/
    │   ├── ARG_CRO_220001.npy    # (NUM_FRAMES, NUM_PERSONS, 4)
    │   └── ...
    ├── images/
    │   ├── ARG_CRO_220001/
    │   │   ├── 00000.jpg
    │   │   ├── 00001.jpg
    │   │   └── ...
    │   └── ...
    ├── skel_2d/   (created empty, SAM writes here)
    └── skel_3d/   (created empty, SAM writes here)

Usage (Colab):
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Option 1: Keypoints and videos in SEPARATE folders
    !python extract_boxes.py \\
        --keypoints_dir /content/drive/MyDrive/keypoints \\
        --videos_dir /content/drive/MyDrive/videos \\
        --output_dir /content/data

    # Option 2: Everything in ONE folder (use same path for both)
    !python extract_boxes.py \\
        --keypoints_dir /content/drive/MyDrive/all_data \\
        --videos_dir /content/drive/MyDrive/all_data \\
        --output_dir /content/data

    # Then run SAM
    !python preprocess.py
"""

import argparse
import os
from pathlib import Path
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================
# 1. BOUNDING BOX EXTRACTION
# ============================================================

def extract_boxes_from_keypoints(j2d, padding=20, image_w=1920, image_h=1080):
    """
    Compute bounding boxes from 2D keypoints.

    Args:
        j2d: np.ndarray of shape (NUM_PERSONS, NUM_FRAMES, K, 2)
        padding: pixels to pad around the skeleton
        image_w: image width for clamping
        image_h: image height for clamping

    Returns:
        boxes: np.ndarray of shape (NUM_FRAMES, NUM_PERSONS, 4) — [x1, y1, x2, y2]
    """
    with np.errstate(all="ignore"):
        x_min = np.nanmin(j2d[..., 0], axis=-1) - padding
        y_min = np.nanmin(j2d[..., 1], axis=-1) - padding
        x_max = np.nanmax(j2d[..., 0], axis=-1) + padding
        y_max = np.nanmax(j2d[..., 1], axis=-1) + padding

    x_min = np.clip(x_min, 0, image_w)
    y_min = np.clip(y_min, 0, image_h)
    x_max = np.clip(x_max, 0, image_w)
    y_max = np.clip(y_max, 0, image_h)

    # Stack: (persons, frames, 4) then transpose to (frames, persons, 4)
    boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)
    boxes = boxes.transpose(1, 0, 2)

    # Set NaN persons (invisible) to [0, 0, 0, 0]
    nan_mask = np.all(np.isnan(j2d), axis=(-1, -2))  # (persons, frames)
    nan_mask = nan_mask.T  # (frames, persons)
    boxes[nan_mask] = 0.0

    return boxes


def process_keypoints(keypoints_path, output_boxes_dir, padding=20):
    """Extract boxes from a single keypoints file and save."""
    data = np.load(keypoints_path, allow_pickle=True)
    j2d = data["joints_2d_pixels"]  # (persons, frames, K, 2)
    seq_name = keypoints_path.stem.replace("_15keypoints_FULL", "")

    boxes = extract_boxes_from_keypoints(j2d, padding=padding)
    output_path = output_boxes_dir / f"{seq_name}.npy"
    np.save(output_path, boxes)

    num_persons, num_frames = j2d.shape[0], j2d.shape[1]
    visible_per_frame = np.mean(np.sum(~np.all(boxes == 0, axis=-1), axis=-1))

    return seq_name, num_frames, num_persons, visible_per_frame


# ============================================================
# 2. VIDEO FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, output_folder):
    """Extract all frames from a video as JPG images."""
    output_folder.mkdir(parents=True, exist_ok=True)

    if HAS_CV2:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if HAS_PIL:
                output_filename = output_folder / f"{frame_count:05d}.jpg"
                Image.fromarray(frame[..., ::-1]).save(str(output_filename), optimize=True)
            else:
                output_filename = output_folder / f"{frame_count:05d}.jpg"
                cv2.imwrite(str(output_filename), frame)
            frame_count += 1
        cap.release()
        return frame_count
    else:
        raise ImportError("cv2 (opencv-python) is required for frame extraction. "
                          "Install with: pip install opencv-python")


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for SAM-3D Body: extract boxes from keypoints + frames from videos"
    )
    parser.add_argument("--keypoints_dir", type=str, required=True,
                        help="Directory containing *_15keypoints_FULL.npz files")
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Directory containing *.mp4 video files")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output root directory (SAM data/ structure)")
    parser.add_argument("--padding", type=int, default=20,
                        help="Padding in pixels around skeleton bounding box")
    parser.add_argument("--skip_frames", action="store_true",
                        help="Skip video frame extraction (only extract boxes)")
    parser.add_argument("--skip_boxes", action="store_true",
                        help="Skip box extraction (only extract frames)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip sequences that already have outputs (default: True)")
    args = parser.parse_args()

    keypoints_dir = Path(args.keypoints_dir)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    # Create output directory structure
    boxes_dir = output_dir / "boxes"
    images_dir = output_dir / "images"
    skel_2d_dir = output_dir / "skel_2d"
    skel_3d_dir = output_dir / "skel_3d"

    boxes_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    skel_2d_dir.mkdir(parents=True, exist_ok=True)
    skel_3d_dir.mkdir(parents=True, exist_ok=True)

    # Find all keypoints files
    npz_files = sorted(keypoints_dir.glob("*_15keypoints_FULL.npz"))
    print(f"=" * 60)
    print(f"SAM Data Preparation Pipeline")
    print(f"=" * 60)
    print(f"Keypoints dir:  {keypoints_dir}")
    print(f"Videos dir:     {videos_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Keypoints files found: {len(npz_files)}")
    print(f"Padding: {args.padding}px")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Skip boxes: {args.skip_boxes}")
    print(f"=" * 60)

    if not npz_files:
        print(f"ERROR: No *_15keypoints_FULL.npz files found in {keypoints_dir}")
        return

    processed_sequences = []
    errors = []

    for i, npz_file in enumerate(npz_files):
        seq_name = npz_file.stem.replace("_15keypoints_FULL", "")
        print(f"\n[{i+1}/{len(npz_files)}] Processing: {seq_name}")

        # --- Extract bounding boxes ---
        if not args.skip_boxes:
            box_path = boxes_dir / f"{seq_name}.npy"
            if args.skip_existing and box_path.exists():
                print(f"  [BOXES] Skipping (already exists): {box_path}")
            else:
                try:
                    seq, nf, np_, vis = process_keypoints(npz_file, boxes_dir, args.padding)
                    print(f"  [BOXES] {nf} frames, {np_} persons, "
                          f"avg {vis:.1f} visible/frame → {box_path}")
                except Exception as e:
                    print(f"  [BOXES] ERROR: {e}")
                    errors.append((seq_name, "boxes", str(e)))

        # --- Extract video frames ---
        if not args.skip_frames:
            img_dir = images_dir / seq_name
            video_path = videos_dir / f"{seq_name}.mp4"

            if args.skip_existing and img_dir.exists() and any(img_dir.iterdir()):
                existing = len(list(img_dir.glob("*.jpg")))
                print(f"  [FRAMES] Skipping (already exists, {existing} frames): {img_dir}")
            elif not video_path.exists():
                print(f"  [FRAMES] WARNING: No video file found: {video_path}")
                # Try other extensions
                for ext in [".MP4", ".avi", ".mov", ".mkv"]:
                    alt = videos_dir / f"{seq_name}{ext}"
                    if alt.exists():
                        video_path = alt
                        print(f"  [FRAMES] Found alternative: {video_path}")
                        break
                else:
                    errors.append((seq_name, "frames", f"Video not found: {video_path}"))
                    continue
            if video_path.exists():
                try:
                    frame_count = extract_frames(video_path, img_dir)
                    print(f"  [FRAMES] {frame_count} frames extracted → {img_dir}")
                except Exception as e:
                    print(f"  [FRAMES] ERROR: {e}")
                    errors.append((seq_name, "frames", str(e)))

        processed_sequences.append(seq_name)

    # --- Generate sequences_full.txt ---
    seq_file = output_dir / "sequences_full.txt"
    with open(seq_file, "w") as f:
        for seq in sorted(processed_sequences):
            f.write(seq + "\n")
    print(f"\n{'=' * 60}")
    print(f"sequences_full.txt written with {len(processed_sequences)} sequences")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed: {len(processed_sequences)} sequences")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for seq, step, err in errors:
            print(f"  {seq} [{step}]: {err}")

    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── sequences_full.txt  ({len(processed_sequences)} sequences)")
    print(f"  ├── boxes/              (NUM_FRAMES, NUM_PERSONS, 4) .npy files")
    print(f"  ├── images/             Extracted video frames as .jpg")
    print(f"  ├── skel_2d/            (empty, SAM writes here)")
    print(f"  └── skel_3d/            (empty, SAM writes here)")


if __name__ == "__main__":
    main()
