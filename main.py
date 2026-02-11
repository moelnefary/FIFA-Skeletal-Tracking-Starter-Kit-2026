"""
FIFA Skeletal Tracking — Team Mil Solution

Upgrades over the baseline:
  • Step 2: Global orientation compensation (R_corr) before camera rotation.
  • Step 4: Tangent-space (se3 / Lie-algebra) reprojection optimisation.

Author: Tianjian Jiang (baseline) — upgraded to Team Mil solution.
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from lib.camera_tracker import CameraTracker, CameraTrackerOptions
from lib.postprocess import smoothen


OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def intersection_over_plane(o, d):
    """
    args:
        o: (3,) origin of the ray
        d: (3,) direction of the ray
    returns:
        intersection: (3,) intersection point with z=0 plane
    """
    t = -o[2] / d[2]
    return o + t * d


def ray_from_xy(xy, K, R, t, k1=0.0, k2=0.0):
    """
    Compute the ray from the camera center through the image point (x, y),
    correcting for radial distortion using coefficients k1 and k2.
    """
    p = np.array([xy[0], xy[1], 1.0])
    p_norm = np.linalg.inv(K) @ p
    x_d, y_d = p_norm[0], p_norm[1]

    r2 = x_d**2 + y_d**2
    factor = 1 + k1 * r2 + k2 * (r2**2)

    x_undist = x_d / factor
    y_undist = y_d / factor

    d_cam = np.array([x_undist, y_undist, 1.0])
    direction = R.T @ d_cam
    direction = direction / np.linalg.norm(direction)

    origin = -R.T @ t
    return origin, direction


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Global Orientation Compensation
# ═══════════════════════════════════════════════════════════════════════════

def compute_orientation_correction(box: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute a rotation matrix that corrects weak-perspective distortion.

    Aligns the camera's optical axis A = [0, 0, 1] to the ray D that passes
    through the centre of the player's 2D bounding box.

    Args:
        box: (4,) bounding box [x1, y1, x2, y2].
        K:   (3, 3) camera intrinsic matrix.

    Returns:
        R_corr: (3, 3) correction rotation matrix.
    """
    # Centre of the bounding box in pixel coordinates
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    # Back-project to get the ray direction in camera coordinates
    D = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
    D = D / np.linalg.norm(D)

    # Optical axis
    A = np.array([0.0, 0.0, 1.0])

    # Rodrigues formula: compute rotation from A to D
    v = np.cross(A, D)
    s = np.linalg.norm(v)
    c = np.dot(A, D)

    if s < 1e-8:
        # A and D are (anti-)parallel — no correction needed
        return np.eye(3)

    vx = np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])

    R_corr = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R_corr


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Tangent-Space (se3) Optimisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def rodrigues_torch(omega: torch.Tensor) -> torch.Tensor:
    """Batch Rodrigues formula: rotation vector → rotation matrix.

    Args:
        omega: (N, 3) rotation vectors.

    Returns:
        R: (N, 3, 3) rotation matrices.
    """
    theta = torch.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)  # (N, 1, 1)
    omega_n = omega / (theta.squeeze(-1) + 1e-8)                  # (N, 3)

    # Skew-symmetric matrix
    zero = torch.zeros_like(omega_n[:, 0])
    K = torch.stack([
        zero,          -omega_n[:, 2],  omega_n[:, 1],
        omega_n[:, 2],  zero,          -omega_n[:, 0],
        -omega_n[:, 1], omega_n[:, 0],  zero,
    ], dim=-1).reshape(-1, 3, 3)

    # Rodrigues: R = I + sin(θ) K + (1 - cos(θ)) K²
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def project_points_th(obj_pts, R, C, K, k):
    """Projects 3D points onto 2D image plane using camera intrinsics and distortion.

    args:
        obj_pts: (N, 3) - 3D points in world space
        R: (3, 3) or (N, 3, 3) - Rotation matrix
        C: (3,) or (N, 3) - Camera center
        K: (3, 3) or (N, 3, 3) - Camera intrinsic matrix
        k: (2,) or (N, 2) - Distortion coefficients

    returns:
        img_pts: (N, 2) - Projected 2D points
    """
    pts_c = (R @ ((obj_pts - C).unsqueeze(-1))).squeeze(-1)
    img_pts = pts_c[:, :2] / pts_c[:, 2:]

    r2 = (img_pts**2).sum(dim=-1, keepdim=True)
    r2 = torch.clamp(r2, 0, 0.5 / min(max(torch.abs(k).max().item(), 1.0), 1.0))
    p = torch.arange(1, k.shape[-1] + 1, device=k.device)
    img_pts = img_pts * (torch.ones_like(r2) + (k * r2.pow(p)).sum(-1, keepdim=True))

    img_pts_h = torch.cat([img_pts, torch.ones_like(img_pts[:, :1])], dim=-1)
    img_pts = (K @ img_pts_h.unsqueeze(-1)).squeeze(-1)[:, :2]
    return img_pts


def minimize_reprojection_error(pts_3d, pts_2d, R, C, K, k, iterations=10):
    """Optimise camera pose in tangent space (se3 / Lie algebra).

    Optimises a 6D twist ξ = [ω (3), δt (3)] per sample so that:
        R_new = R_current @ exp(ω)      (rotation update via Rodrigues)
        t_new = t_current + δt           (translation update)
    minimising the MSE between projected 3D hip joints and 2D detections.

    Args:
        pts_3d: (N, 3)   Initial 3D points.
        pts_2d: (N, 2)   Corresponding 2D detections.
        R:      (N, 3, 3) Rotation matrices (fixed reference).
        C:      (N, 3)    Camera centres (fixed reference).
        K:      (N, 3, 3) Intrinsic matrices.
        k:      (N, 2)    Distortion coefficients.
        iterations: int   Number of L-BFGS steps.

    Returns:
        omega:  (N, 3)  Optimised rotation deltas (as rotation vectors).
        delta_t: (N, 3) Optimised translation deltas.
    """
    N = pts_3d.shape[0]
    device = pts_3d.device
    dtype = pts_3d.dtype

    # Learnable se3 parameters: rotation vector ω and translation δt
    omega = torch.nn.Parameter(torch.zeros(N, 3, device=device, dtype=dtype))
    delta_t = torch.nn.Parameter(torch.zeros(N, 3, device=device, dtype=dtype))

    # Bounds for clamping
    omega_bound = 0.1   # ~5.7 degrees max rotation update
    t_bound = torch.tensor([3.0, 3.0, 0.2], device=device, dtype=dtype)

    assert not torch.isnan(pts_3d).any()
    assert not torch.isnan(pts_2d).any()

    def closure():
        optimizer.zero_grad()

        # Compute updated rotation: R_new = R_current @ exp(ω)
        dR = rodrigues_torch(omega)          # (N, 3, 3)
        R_new = R @ dR                       # (N, 3, 3)

        # Updated camera centre: C_new = C + δt
        C_new = C + delta_t

        projected = project_points_th(pts_3d, R_new, C_new, K, k)
        loss = torch.nn.functional.mse_loss(projected, pts_2d)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([omega, delta_t], line_search_fn="strong_wolfe")
    for _ in range(iterations):
        optimizer.step(closure)
        with torch.no_grad():
            omega.clamp_(-omega_bound, omega_bound)
            delta_t.copy_(torch.clamp(delta_t, -t_bound, t_bound))

    return omega.detach(), delta_t.detach()


def fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes):
    """Wrapper to fine-tune the 3D predictions using tangent-space optimisation."""
    NUM_PERSONS = predictions.shape[0]
    mid_hip_3d = predictions[..., [7, 8], :].mean(axis=-2, keepdims=False)
    mid_hip_2d = skels_2d[..., [7, 8], :].mean(axis=-2, keepdims=False).transpose(1, 0, 2)

    R = np.array([k[0] for k in Rt])
    t = np.array([k[1] for k in Rt])
    C = (-t[:, None] @ R).squeeze(1)

    camera_params = {
        "K": cameras["K"][None].repeat(NUM_PERSONS, axis=0),
        "R": R[None].repeat(NUM_PERSONS, axis=0),
        "C": C[None].repeat(NUM_PERSONS, axis=0),
        "k": cameras["k"][None, ..., :2].repeat(NUM_PERSONS, axis=0),
    }
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)

    omega, delta_t = minimize_reprojection_error(
        pts_3d=torch.tensor(mid_hip_3d[valid], dtype=torch.float32).to("cuda"),
        pts_2d=torch.tensor(mid_hip_2d[valid], dtype=torch.float32).to("cuda"),
        R=torch.tensor(camera_params["R"][valid], dtype=torch.float32).to("cuda"),
        C=torch.tensor(camera_params["C"][valid], dtype=torch.float32).to("cuda"),
        K=torch.tensor(camera_params["K"][valid], dtype=torch.float32).to("cuda"),
        k=torch.tensor(camera_params["k"][valid], dtype=torch.float32).to("cuda"),
    )
    return omega, delta_t, valid


# ═══════════════════════════════════════════════════════════════════════════
# Main processing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_sequence(
    boxes: np.ndarray,
    cameras: dict,
    skels_3d: np.ndarray,
    skels_2d: np.ndarray,
    video_path: Path | str,
    tracker_options: CameraTrackerOptions,
) -> np.ndarray:
    """Process one video sequence.

    1. Estimate/track camera pose per frame (PnP-based tracker).
    2. For each person, apply global orientation compensation (R_corr)
       then camera extrinsics to lift 3D skeletons into world space.
    3. Fine-tune via tangent-space reprojection optimisation.
    """
    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    predictions = np.zeros((NUM_PERSONS, NUM_FRAMES, 15, 3))
    predictions.fill(np.nan)
    pitch_points = np.loadtxt("data/pitch_points.txt")

    video = cv2.VideoCapture(video_path)
    camera_tracker = CameraTracker(
        pitch_points=pitch_points,
        fps=50.0,
        options=tracker_options,
    )
    camera_tracker.initialize(
        frame_idx=0,
        K=cameras["K"][0],
        k=cameras["k"][0],
        R=cameras["R"][0],
        t=cameras["t"][0],
    )

    Rt = []
    for frame_idx in (pbar := tqdm(range(NUM_FRAMES), desc=f"{video_path.stem}")):
        success, img = video.read()
        if not success:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            break

        state = camera_tracker.track(
            frame_idx=frame_idx,
            frame=img,
            K=cameras["K"][frame_idx],
            dist_coeffs=cameras["k"][frame_idx],
        )
        yaw, pitch, roll = state.get_ypr()
        pbar.set_postfix_str(f"yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
        Rt.append((state.R.copy(), state.t.copy()))

        for person in range(NUM_PERSONS):
            box = boxes[frame_idx, person]
            if np.isnan(box).any():
                continue

            skel_2d = skels_2d[frame_idx, person]

            IDX = np.argmax(skel_2d[:, 1])
            x, y = skel_2d[IDX]
            K = cameras["K"][frame_idx]
            k = cameras["k"][frame_idx]
            R, t = Rt[-1]
            o, d = ray_from_xy((x, y), K, R, t, k[0], k[1])
            intersection = intersection_over_plane(o, d)

            # ── Step 2: Global orientation compensation ──────────────
            R_corr = compute_orientation_correction(box, K)

            skel_3d = skels_3d[frame_idx, person]
            skel_3d = (skel_3d @ R_corr) @ R          # corrected rotation
            skel_3d = skel_3d - skel_3d[IDX] + intersection
            predictions[person, frame_idx] = skel_3d

    # ── Step 4: Tangent-space fine-tuning ────────────────────────────────
    omega, delta_t, valid = fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes)

    # Apply the optimised rotation and translation deltas
    dR = rodrigues_torch(omega).cpu().numpy()           # (M, 3, 3)
    dt_np = delta_t.cpu().numpy()                       # (M, 3)

    # Rotate each skeleton by dR and shift by δt
    preds_valid = predictions[valid]                    # (M, 15, 3)
    for i in range(len(preds_valid)):
        preds_valid[i] = preds_valid[i] @ dR[i].T      # apply rotation delta
    preds_valid = preds_valid + dt_np[:, None, :]       # apply translation delta
    predictions[valid] = preds_valid

    for person in range(NUM_PERSONS):
        predictions[person] = smoothen(predictions[person])

    # Update camera parameters
    cameras["R"] = np.array([k[0] for k in Rt], dtype=np.float32)
    cameras["t"] = np.array([k[1] for k in Rt], dtype=np.float32)
    return predictions.astype(np.float32)


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file) as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences


def main(
    sequences: list[str],
    output: Path | str,
    max_refine_interval: int,
    export_camera: bool,
    visualize: bool,
):
    debug_stages = ["projection", "flow", "mask"] if visualize else []
    if export_camera:
        camera_dir = Path("outputs/calibration/")
        camera_dir.mkdir(parents=True, exist_ok=True)
    else:
        camera_dir = None

    root = Path("data/")
    solutions = {}
    for sequence in sequences:
        camera = dict(np.load(root / "cameras" / f"{sequence}.npz"))
        skel2d = np.load(root / "skel_2d" / f"{sequence}.npy")
        skel3d = np.load(root / "skel_3d" / f"{sequence}.npy")
        boxes = np.load(root / "boxes" / f"{sequence}.npy")
        video_path = root / "videos" / f"{sequence}.mp4"

        NUM_FRAMES = boxes.shape[0]
        solutions[sequence] = process_sequence(
            cameras=camera,
            boxes=boxes,
            skels_2d=skel2d[:, :, OPENPOSE_TO_OURS],
            skels_3d=skel3d[:, :, OPENPOSE_TO_OURS],
            video_path=video_path,
            tracker_options=CameraTrackerOptions(
                refine_interval=np.clip(NUM_FRAMES // 500, a_min=1, a_max=max_refine_interval),
                debug_stages=tuple(debug_stages),
            ),
        )

        if export_camera:
            camera_path = camera_dir / f"{sequence}.npz"
            np.savez(camera_path, **camera)

    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **solutions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences", "-s", type=str, default="data/sequences_full.txt", help="Path to the sequences file"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default="output/submission_full.npz", help="Path to the output npz file"
    )
    parser.add_argument("--refine_interval", "-r", type=int, default=1, help="Interval to refine the camera pose")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the tracking results")
    parser.add_argument("--export_camera", "-c", action="store_true", help="Export the camera parameters")
    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    main(sequences, args.output, args.refine_interval, args.export_camera, args.visualize)
