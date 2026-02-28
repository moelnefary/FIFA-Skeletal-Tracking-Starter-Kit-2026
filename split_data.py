"""
Split SAM data into N parts for parallel processing on separate Colab sessions.

Takes a folder containing boxes/, images/, cameras/ and splits it into
N self-contained part folders. Uses symlinks (Linux/Colab) to avoid duplicating
the large image/box/camera data.

Each part folder has:
  part_X/
  ├── sequences_full.txt  (only this part's sequences)
  ├── boxes/    → symlink to shared data
  ├── images/   → symlink to shared data
  ├── cameras/  → symlink to shared data
  ├── skel_2d/  (empty, SAM writes here)
  └── skel_3d/  (empty, SAM writes here)

Usage (Colab):
    !python split_data.py \
        --data_dir /content/data \
        --output_dir /content/parts \
        --num_parts 4

Then each team member runs:
    !python preprocess_colab.py --data_dir /content/parts/part_1
    !python preprocess_colab.py --data_dir /content/parts/part_2
    !python preprocess_colab.py --data_dir /content/parts/part_3
    !python preprocess_colab.py --data_dir /content/parts/part_4
"""

import argparse
from pathlib import Path
import os
import numpy as np


def estimate_time_per_seq(boxes_dir, seq_name):
    """Estimate processing time for a sequence on T4 GPU."""
    boxes_path = boxes_dir / f"{seq_name}.npy"
    if not boxes_path.exists():
        return 0, 0, 0

    boxes = np.load(boxes_path)
    num_frames = boxes.shape[0]
    num_persons = boxes.shape[1]

    # Count average visible persons per frame (non-zero boxes)
    visible_per_frame = np.mean(np.sum(~np.all(boxes == 0, axis=-1), axis=-1))

    # Estimate: ~0.07 sec per person per frame on T4
    est_seconds = num_frames * visible_per_frame * 0.07

    return num_frames, visible_per_frame, est_seconds


def main():
    parser = argparse.ArgumentParser(description="Split SAM data into parts for parallel Colab sessions")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data dir with boxes/, images/, cameras/, sequences_full.txt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for part folders")
    parser.add_argument("--num_parts", type=int, default=4,
                        help="Number of parts to split into (default: 4)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Load sequences
    seq_file = data_dir / "sequences_full.txt"
    if not seq_file.exists():
        print(f"ERROR: {seq_file} not found")
        return

    with open(seq_file) as f:
        all_seqs = [s.strip() for s in f.readlines() if s.strip() and not s.startswith("#")]

    print("=" * 70)
    print(f"Splitting {len(all_seqs)} sequences into {args.num_parts} parts")
    print("=" * 70)

    # Estimate time for each sequence
    seq_times = []
    total_frames = 0
    for seq in all_seqs:
        nf, avg_vis, est_sec = estimate_time_per_seq(data_dir / "boxes", seq)
        seq_times.append((seq, nf, avg_vis, est_sec))
        total_frames += nf

    # Sort by estimated time (heaviest first) for better load balancing
    seq_times.sort(key=lambda x: -x[3])

    # Distribute sequences using round-robin on sorted list (balanced by time)
    parts = [[] for _ in range(args.num_parts)]
    part_times = [0.0] * args.num_parts
    part_frames = [0] * args.num_parts

    for seq, nf, avg_vis, est_sec in seq_times:
        # Assign to the part with least total time so far
        min_part = min(range(args.num_parts), key=lambda i: part_times[i])
        parts[min_part].append(seq)
        part_times[min_part] += est_sec
        part_frames[min_part] += nf

    # Create part folders
    shared_dirs = ["boxes", "images", "cameras"]

    for i, part_seqs in enumerate(parts):
        part_num = i + 1
        part_dir = output_dir / f"part_{part_num}"
        part_dir.mkdir(parents=True, exist_ok=True)

        # Write this part's sequences_full.txt
        with open(part_dir / "sequences_full.txt", "w") as f:
            f.write("\n".join(part_seqs) + "\n")

        # Create symlinks to shared data dirs
        for dirname in shared_dirs:
            link_path = part_dir / dirname
            target = data_dir / dirname
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    # Already exists as a real dir, skip
                    continue
            try:
                os.symlink(str(target), str(link_path))
            except OSError:
                # Windows or permission issue — fall back to just noting it
                print(f"  WARNING: Could not create symlink {link_path} -> {target}")
                print(f"           On Colab (Linux), symlinks will work fine.")

        # Create output dirs
        (part_dir / "skel_2d").mkdir(exist_ok=True)
        (part_dir / "skel_3d").mkdir(exist_ok=True)

        # Print part info
        est_hours = part_times[i] / 3600
        print(f"\n{'─' * 70}")
        print(f"Part {part_num}: {len(part_seqs)} sequences, "
              f"{part_frames[i]} total frames")
        print(f"  Estimated time on T4: ~{est_hours:.1f} hours")
        print(f"  Sequences: {', '.join(part_seqs[:5])}"
              f"{'...' if len(part_seqs) > 5 else ''}")
        print(f"  Path: {part_dir}")

    # Summary
    total_est = sum(part_times) / 3600
    max_est = max(part_times) / 3600
    min_est = min(part_times) / 3600

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total sequences:  {len(all_seqs)}")
    print(f"Total frames:     {total_frames}")
    print(f"Total est. time:  ~{total_est:.1f} hours (sequential)")
    print(f"Per part:         ~{min_est:.1f} - {max_est:.1f} hours (parallel on T4)")
    print(f"\nData structure per part:")
    print(f"  part_X/")
    print(f"  ├── sequences_full.txt  (this part's sequences)")
    print(f"  ├── boxes/    → {data_dir / 'boxes'}")
    print(f"  ├── images/   → {data_dir / 'images'}")
    print(f"  ├── cameras/  → {data_dir / 'cameras'}")
    print(f"  ├── skel_2d/  (SAM output)")
    print(f"  └── skel_3d/  (SAM output)")
    print(f"\nCommands to run:")
    for i in range(args.num_parts):
        print(f"  Team {i+1}: python preprocess_colab.py "
              f"--data_dir {output_dir / f'part_{i+1}'}")
    print(f"\nAfter all parts finish, collect outputs:")
    print(f"  All skel_2d/*.npy and skel_3d/*.npy from each part_X/ folder")


if __name__ == "__main__":
    main()
