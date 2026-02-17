
"""
This script extracts 2D and 3D keypoints from 2D detections using the 4DHuman model.
Please refer to the https://github.com/shubham-goel/4D-Humans/tree/main for installation instructions.

Author: Tianjian Jiang
Date: March 16, 2025
"""
from pathlib import Path
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

def load_sequences(root):
    with open(root / "sequences_full.txt", "r") as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences

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

def main(root):
    model = SAM3D("cuda")
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