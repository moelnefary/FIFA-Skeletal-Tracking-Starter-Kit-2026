
"""
Colab-compatible SAM-3D Body preprocessing script.
Splits sequences into 4 parts for parallel processing across team members.

Each team member runs on a separate Colab instance with their assigned part.
After all 4 finish, collect outputs from data/skel_2d/ and data/skel_3d/.

Usage (Colab):
    # Option 1: After running split_data.py (each member gets a pre-split folder):
    !python preprocess_colab.py --data_dir /content/parts/part_1
    !python preprocess_colab.py --data_dir /content/parts/part_2

    # Option 2: Manual split with --part flag (all members share same data dir):
    !python preprocess_colab.py --data_dir /content/data --part 1 --total_parts 4
    !python preprocess_colab.py --data_dir /content/data --part 2 --total_parts 4

Author: Tianjian Jiang (modified for parallel Colab execution)
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from tqdm import trange
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf


def run_eval(model, image_dir, boxes, cam_int=None):
    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    skels_2d = np.zeros((NUM_FRAMES, NUM_PERSONS, 25, 2))
    skels_3d = np.zeros((NUM_FRAMES, NUM_PERSONS, 25, 3))
    skels_2d.fill(np.nan)
    skels_3d.fill(np.nan)

    image_files = sorted(list(image_dir.glob("*.jpg")))
    for frame in (pbar := trange(NUM_FRAMES, desc=f"{image_dir.stem}")):
        img = image_files[frame]
        skels_2d[frame], skels_3d[frame] = model(img, boxes[frame], cam_int=cam_int[frame])
    return skels_2d, skels_3d


def load_sequences(root, seq_file="sequences_full.txt"):
    with open(root / seq_file, "r") as f:
        sequences = f.read().splitlines()
    sequences = [s.strip() for s in sequences if s.strip() and not s.startswith("#")]
    return sequences


def split_sequences(sequences, part, total_parts):
    """Split sequences into equal parts. Returns the sequences for the given part."""
    n = len(sequences)
    chunk_size = n // total_parts
    remainder = n % total_parts

    # Distribute remainder across first 'remainder' parts
    start = 0
    for i in range(1, part):
        start += chunk_size + (1 if i <= remainder else 0)
    end = start + chunk_size + (1 if part <= remainder else 0)

    return sequences[start:end]


class SAM3D:
    """A wrapper around the SAM3D model to extract 3D keypoints from 2D detections."""
    def __init__(self, device):
        model, model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-dinov3")
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
        )

    def sam3d_to_body25(self, kpt):
        """for backward compatibility with the openpose format"""
        INDICES_70_TO_BODY25 = [
            0,    # 0 Nose
            69,   # 1 Neck
            6,    # 2 RShoulder
            8,    # 3 RElbow
            41,   # 4 RWrist
            5,    # 5 LShoulder
            7,    # 6 LElbow
            62,   # 7 LWrist
            -1,   # 8 MidHip  (compute as avg of indices 9 and 10)
            10,   # 9 RHip
            12,   # 10 RKnee
            14,   # 11 RAnkle
            9,    # 12 LHip
            11,   # 13 LKnee
            13,   # 14 LAnkle
            2,    # 15 REye
            1,    # 16 LEye
            4,    # 17 REar
            3,    # 18 LEar
            15,   # 19 LBigToe
            16,   # 20 LSmallToe
            17,   # 21 LHeel
            18,   # 22 RBigToe
            19,   # 23 RSmallToe
            20,   # 24 RHeel
        ]
        kp25 = kpt[..., INDICES_70_TO_BODY25, :]
        kp25[..., 8, :] = (kpt[..., 9, :] + kpt[..., 10, :]) / 2
        return kp25

    def __call__(self, img, boxes=None, cam_int=None):
        """
        args:
            img: (H, W, 3) in RGB format or str
        """
        if isinstance(img, Path): img = str(img)
        if cam_int is not None:
            if isinstance(cam_int, np.ndarray):
                cam_int = torch.from_numpy(cam_int).float().to(self.estimator.device)
            cam_int = cam_int.reshape(1, 3, 3)
        batch = self.estimator.process_one_image(
            img, bboxes=boxes, cam_int=cam_int,
            inference_type="body"
        )
        assert len(batch) == len(boxes), "Number of boxes and batch should be the same"
        kpt_2d = np.zeros((len(boxes), 70, 2))
        kpt_3d = np.zeros((len(boxes), 70, 3))
        for person_id, pitem in enumerate(batch):
            kpt_2d[person_id] = pitem["pred_keypoints_2d"]
            kpt_3d[person_id] = pitem["pred_keypoints_3d"]
        kpt_2d = self.sam3d_to_body25(kpt_2d)
        kpt_3d = self.sam3d_to_body25(kpt_3d)
        return kpt_2d, kpt_3d


def main():
    parser = argparse.ArgumentParser(description="SAM-3D Body preprocessing (Colab-ready, parallelizable)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory containing boxes/, images/, cameras/, skel_2d/, skel_3d/")
    parser.add_argument("--part", type=int, default=None,
                        help="Which part to process (1-indexed). If not set, processes ALL sequences in data_dir")
    parser.add_argument("--total_parts", type=int, default=4,
                        help="Total number of parts to split into (default: 4)")
    args = parser.parse_args()

    root = Path(args.data_dir)

    # Create output dirs
    (root / "skel_2d").mkdir(parents=True, exist_ok=True)
    (root / "skel_3d").mkdir(parents=True, exist_ok=True)

    # Load and split sequences
    all_sequences = load_sequences(root)

    if args.part is not None:
        my_sequences = split_sequences(all_sequences, args.part, args.total_parts)
        part_label = f"Part {args.part}/{args.total_parts}"
    else:
        my_sequences = all_sequences
        part_label = "All sequences"

    print("=" * 60)
    print(f"SAM-3D Body Preprocessing — {part_label}")
    print("=" * 60)
    print(f"Total sequences in file: {len(all_sequences)}")
    print(f"Processing: {len(my_sequences)} sequences")
    print(f"Sequences: {my_sequences}")
    print("=" * 60)

    # Load model
    print("Loading SAM-3D Body model...")
    model = SAM3D("cuda")
    print("Model loaded!")

    completed = 0
    skipped = 0
    errors = []

    for i, seq in enumerate(my_sequences):
        skel_2d_path = root / "skel_2d" / f"{seq}.npy"
        skel_3d_path = root / "skel_3d" / f"{seq}.npy"

        # Skip if already done
        if skel_2d_path.exists() and skel_3d_path.exists():
            print(f"[{i+1}/{len(my_sequences)}] {seq}: SKIPPED (already exists)")
            skipped += 1
            continue

        try:
            camera = np.load(root / "cameras" / f"{seq}.npz")
            cam_int = camera["K"]
            boxes = np.load(root / "boxes" / f"{seq}.npy")
            image_dir = root / "images" / seq

            print(f"\n[{i+1}/{len(my_sequences)}] {seq}: "
                  f"{boxes.shape[0]} frames, {boxes.shape[1]} persons")

            skel_2d, skel_3d = run_eval(model, image_dir, boxes, cam_int)

            np.save(skel_2d_path, skel_2d)
            np.save(skel_3d_path, skel_3d)
            completed += 1
            print(f"  → Saved: {skel_2d_path}, {skel_3d_path}")

        except Exception as e:
            print(f"  → ERROR: {e}")
            errors.append((seq, str(e)))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DONE — {part_label}")
    print(f"{'=' * 60}")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {len(errors)}")
    if errors:
        for seq, err in errors:
            print(f"  {seq}: {err}")


if __name__ == "__main__":
    main()
