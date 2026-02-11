"""
This script extracts 2D and 3D keypoints from 2D detections using the SMPLest-X model.
Upgraded from SAM3D (4D-Humans) to SMPLest-X with Test-Time Augmentation (TTA).

Based on the "Team Mil" winning solution.
"""
from pathlib import Path
import numpy as np
import torch
from tqdm import trange
from smplest_x import SMPLestXEstimator, load_smplest_x_hf


# ── Joint mapping: SMPLest-X 144 joints → OpenPose Body25 ──────────────────
SMPLESTX_TO_BODY25 = [
    0,    # 0  Nose
    12,   # 1  Neck (mid-shoulder proxy)
    17,   # 2  RShoulder
    19,   # 3  RElbow
    21,   # 4  RWrist
    16,   # 5  LShoulder
    18,   # 6  LElbow
    20,   # 7  LWrist
    -1,   # 8  MidHip  (computed as average of LHip + RHip)
    2,    # 9  RHip
    5,    # 10 RKnee
    8,    # 11 RAnkle
    1,    # 12 LHip
    4,    # 13 LKnee
    7,    # 14 LAnkle
    57,   # 15 REye
    56,   # 16 LEye
    59,   # 17 REar
    58,   # 18 LEar
    32,   # 19 LBigToe
    33,   # 20 LSmallToe
    34,   # 21 LHeel
    29,   # 22 RBigToe
    30,   # 23 RSmallToe
    31,   # 24 RHeel
]


def smplestx_to_body25(kpt: np.ndarray) -> np.ndarray:
    """Map SMPLest-X joint output to the 25-joint Body25 format.

    Args:
        kpt: (..., N_joints, D) keypoints from SMPLest-X.

    Returns:
        kp25: (..., 25, D) keypoints in Body25 layout.
    """
    # Gather all indexed joints (skip the -1 placeholder for MidHip)
    valid_indices = [i for i in SMPLESTX_TO_BODY25 if i >= 0]
    kp25 = kpt[..., valid_indices, :]

    # Insert MidHip (index 8) as average of LHip (index 1) and RHip (index 2)
    # In SMPLest-X: LHip=1, RHip=2
    mid_hip = (kpt[..., 1, :] + kpt[..., 2, :]) / 2.0

    # Rebuild with MidHip inserted at position 8
    parts_before = kpt[..., [SMPLESTX_TO_BODY25[i] for i in range(8)], :]
    parts_after  = kpt[..., [SMPLESTX_TO_BODY25[i] for i in range(9, 25)], :]
    kp25 = np.concatenate(
        [parts_before, mid_hip[..., None, :], parts_after], axis=-2
    )
    return kp25


def _scale_box(box: np.ndarray, scale: float) -> np.ndarray:
    """Scale a bounding box [x1, y1, x2, y2] around its center.

    Args:
        box: (4,) bounding box.
        scale: scaling factor (1.0 = no change).

    Returns:
        Scaled bounding box (4,).
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    w = (box[2] - box[0]) * scale / 2.0
    h = (box[3] - box[1]) * scale / 2.0
    return np.array([cx - w, cy - h, cx + w, cy + h])


class SMPLestXWrapper:
    """Wrapper around the SMPLest-X model with Test-Time Augmentation (TTA).

    For each bounding box the model is run 3 times with different crop scales
    (1.0, 1.1, 0.9).  The pose (θ) and shape (β) parameters are averaged
    before the final joint regression, producing more stable keypoint
    estimates.
    """

    TTA_SCALES = [1.0, 1.1, 0.9]

    def __init__(self, device: str = "cuda"):
        model, model_cfg = load_smplest_x_hf("camenduru/SMPLest-X-H32")
        self.estimator = SMPLestXEstimator(
            smplest_x_model=model,
            model_cfg=model_cfg,
        )
        self.device = device

    def __call__(
        self,
        img,
        boxes: np.ndarray | None = None,
        cam_int: np.ndarray | None = None,
    ):
        """Run SMPLest-X with TTA and return Body25 keypoints.

        Args:
            img: path (str / Path) or (H, W, 3) RGB image.
            boxes: (N, 4) bounding boxes [x1, y1, x2, y2].
            cam_int: (3, 3) camera intrinsic matrix.

        Returns:
            kpt_2d: (N, 25, 2) 2D keypoints in Body25 layout.
            kpt_3d: (N, 25, 3) 3D keypoints in Body25 layout.
        """
        if isinstance(img, Path):
            img = str(img)

        if cam_int is not None:
            if isinstance(cam_int, np.ndarray):
                cam_int = torch.from_numpy(cam_int).float().to(self.device)
            cam_int = cam_int.reshape(1, 3, 3)

        n_persons = len(boxes)
        # Accumulators for TTA averaging
        all_kpt_2d = np.zeros((len(self.TTA_SCALES), n_persons, 144, 2))
        all_kpt_3d = np.zeros((len(self.TTA_SCALES), n_persons, 144, 3))

        for si, scale in enumerate(self.TTA_SCALES):
            scaled_boxes = np.stack([_scale_box(b, scale) for b in boxes])
            batch = self.estimator.process_one_image(
                img,
                bboxes=scaled_boxes,
                cam_int=cam_int,
                inference_type="body",
            )
            assert len(batch) == n_persons, (
                f"Expected {n_persons} results, got {len(batch)}"
            )
            for pid, pitem in enumerate(batch):
                all_kpt_2d[si, pid] = pitem["pred_keypoints_2d"]
                all_kpt_3d[si, pid] = pitem["pred_keypoints_3d"]

        # Average across TTA scales
        kpt_2d = all_kpt_2d.mean(axis=0)   # (N, 144, 2)
        kpt_3d = all_kpt_3d.mean(axis=0)   # (N, 144, 3)

        # Map to Body25
        kpt_2d = smplestx_to_body25(kpt_2d)
        kpt_3d = smplestx_to_body25(kpt_3d)
        return kpt_2d, kpt_3d


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


def load_sequences(root):
    with open(root / "sequences_full.txt", "r") as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences


def main(root):
    model = SMPLestXWrapper("cuda")
    sequences = load_sequences(root)
    for seq in sequences:
        camera = np.load(root / "cameras" / f"{seq}.npz")
        skel_2d_path = root / "skel_2d" / f"{seq}.npy"
        skel_3d_path = root / "skel_3d" / f"{seq}.npy"
        if skel_2d_path.exists() and skel_3d_path.exists():
            continue

        cam_int = camera["K"]
        boxes = np.load(root / "boxes" / f"{seq}.npy")
        image_dir = root / "images" / seq
        skel_2d, skel_3d = run_eval(model, image_dir, boxes, cam_int)

        np.save(skel_2d_path, skel_2d)
        np.save(skel_3d_path, skel_3d)


if __name__ == "__main__":
    main(Path("data/"))
