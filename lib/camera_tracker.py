from dataclasses import dataclass, field
from collections import deque
import numpy as np
import cv2
from typing import Tuple, Optional


class Debugger:
    """
    Centralized visualization manager for debugging camera tracking.
    """

    def __init__(self, debug_stages: Tuple[str, ...] = ("projection",)):
        """
        Args:
            debug_stages: stages to visualize, e.g., ('flow', 'mask')
        """
        self.stages = set(debug_stages)
        self.frame_curr = None

    def update(self, frame):
        self.frame_curr = frame

    @property
    def visualize(self) -> bool:
        return len(self.stages) > 0

    def draw_optical_flow(
        self,
        pts_prev: np.ndarray,
        pts_next: np.ndarray,
        status: np.ndarray,
    ) -> None:
        """Visualize optical flow vectors."""
        if "flow" not in self.stages:
            return
        vis = self.frame_curr
        for i in range(len(pts_prev)):
            if not status[i]:
                continue
            pt1 = tuple(pts_prev[i].astype(int))
            pt2 = tuple(pts_next[i].astype(int))
            cv2.circle(vis, pt1, 5, (0, 0, 255), -1)  # Red: previous
            cv2.circle(vis, pt2, 5, (0, 255, 0), -1)  # Green: current
            cv2.line(vis, pt1, pt2, (0, 255, 255), 1)  # Yellow: flow vector

    def draw_projection(
        self,
        pts_3d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """Visualize 3D points projected onto current frame."""
        if "projection" not in self.stages:
            return
        vis = self.frame_curr
        pts_2d, _ = cv2.projectPoints(pts_3d, cv2.Rodrigues(R)[0], t, K, dist_coeffs)
        pts_2d = pts_2d.reshape(-1, 2)
        for pt in pts_2d:
            max_size = vis.shape[1::-1]
            valid = (pt >= 0).all() & (pt < max_size).all()
            if not valid:
                continue
            center = pt.astype(int)
            bl = (center - np.array([2, 2])).clip(min=0, max=max_size)
            tr = (center + np.array([2, 2])).clip(min=0, max=max_size)
            cv2.rectangle(vis, tuple(bl), tuple(tr), (0, 255, 255), -1)


def optical_flow_pyrlk(prev_frame, frame, pts_old):
    """
    Calculate the optical flow using the PyRLK algorithm.
    Args:
        prev_frame: The previous frame.
        frame: The current frame.
        pts_old: The previous points (N, 2).
    Returns:
        pts_next: The next points.
        status: The status of the points.
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        minEigThreshold=1e-3,
    )

    pts_next, status, errs = cv2.calcOpticalFlowPyrLK(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        pts_old.reshape(-1, 1, 2).astype(np.float32),
        None,
        **lk_params
    )
    pts_next = pts_next.reshape(-1, 2)
    status = status.ravel().astype(bool)

    # filter out the points with large errors with modified z-score
    errs = np.linalg.norm(pts_next - pts_old, axis=-1)
    median = np.median(errs[status])
    d = np.abs(errs[status] - median)
    mad = np.median(d).clip(min=1e-6)
    modified_z_scores = 0.6745 * d / mad
    status[status] = modified_z_scores <= 3.5
    return pts_next, status


@dataclass
class CameraState:
    frame_idx: int
    K: np.ndarray = field(default_factory=lambda: np.eye(3))
    k: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    C: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "CameraState":
        return CameraState(
            frame_idx=self.frame_idx,
            K=self.K.copy(),
            k=self.k.copy(),
            R=self.R.copy(),
            C=self.C.copy(),
        )

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    def get_ypr(self, deg: bool = True) -> Tuple[float, float, float]:
        yaw, pitch, roll = CameraTracker.rotation_matrix_to_euler(self.R)
        if deg:
            return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)
        else:
            return yaw, pitch, roll


# ── PnP solver ──────────────────────────────────────────────────────────────

def solve_camera_pnp(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve camera pose using PnP with an initial guess.

    Tries iterative PnP first; falls back to RANSAC on failure.

    Args:
        pts_3d: (N, 3) 3D world points.
        pts_2d: (N, 2) corresponding 2D image points.
        K: (3, 3) camera intrinsic matrix.
        dist_coeffs: distortion coefficients.
        R_init: (3, 3) initial rotation matrix.
        t_init: (3,) initial translation vector.

    Returns:
        R: (3, 3) refined rotation matrix.
        t: (3,) refined translation vector.
    """
    rvec_init = cv2.Rodrigues(R_init)[0]
    tvec_init = t_init.reshape(3, 1).astype(np.float64)

    pts_3d = np.ascontiguousarray(pts_3d, dtype=np.float64)
    pts_2d = np.ascontiguousarray(pts_2d, dtype=np.float64)

    # Try iterative solvePnP with initial guess
    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, K, dist_coeffs,
        rvec=rvec_init.copy(), tvec=tvec_init.copy(),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        # Fallback: RANSAC
        success, rvec, tvec, _inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, dist_coeffs,
            rvec=rvec_init.copy(), tvec=tvec_init.copy(),
            useExtrinsicGuess=True,
            iterationsCount=200,
            reprojectionError=5.0,
        )

    if not success:
        # PnP completely failed — keep the previous pose
        return R_init, t_init

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


@dataclass
class CameraTrackerOptions:
    refine_interval: int = 10
    debug_stages: Tuple[str, ...] = ("projection",)


class CameraTracker:
    """
    Tracks camera extrinsics using Euler angles (yaw, pitch, roll) and position.

    Designed for broadcast football footage where:
    - Camera position is typically fixed or slowly moving
    - Smooth rotation (pan/tilt) is the primary motion
    - Roll is usually close to zero

    Uses PnP (Perspective-n-Point) with pseudo-landmarks for dead-zone
    stabilisation instead of the older mask-based refinement approach.
    """

    # Minimum number of tracked points before pseudo-landmarks are generated
    MIN_TRACKED_POINTS = 6

    def __init__(
        self,
        pitch_points: np.ndarray,
        fps: float = 30.0,
        options: CameraTrackerOptions = CameraTrackerOptions(),
    ):
        """Initialize the camera tracker."""
        self.state = None
        self.velocity = None
        self.covariance = None
        self.frame_buffer = deque(maxlen=3)
        self.camera_states = []
        self.pitch_points = pitch_points
        self.refine_interval = options.refine_interval
        self.debug_vis = Debugger(debug_stages=options.debug_stages)

    def initialize(self, frame_idx: int, K: np.ndarray, k: np.ndarray, R: np.ndarray, t: np.ndarray) -> None:
        """Initialize tracker from rotation matrix and translation vector."""
        C = -R.T @ t
        self.state = CameraState(frame_idx=frame_idx, K=K, k=k, R=R, C=C)

    def track(self, frame_idx: int, frame: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> CameraState:
        """Track camera pose for the given frame.

        1. Update intrinsics.
        2. Estimate rotation from optical flow.
        3. Refine with PnP (+pseudo-landmarks when tracked points are sparse).
        """
        self.state.frame_idx = frame_idx
        self.state.K = K
        self.state.k = dist_coeffs
        self.debug_vis.update(frame.copy())

        if frame_idx > 0:
            pts_3d_tracked, pts_2d_tracked = self._update_flow(
                frame=frame,
                prev_frame=self.frame_buffer[-1],
                state_prev=self.camera_states[-1],
                state_curr=self.state,
            )

            # ── PnP refinement ──────────────────────────────────────────
            refine = frame_idx % self.refine_interval == 0
            if refine and pts_3d_tracked is not None:
                n_tracked = len(pts_3d_tracked)

                # Dead-zone handling: add pseudo-landmarks when points are few
                if n_tracked < self.MIN_TRACKED_POINTS:
                    pseudo_3d, pseudo_2d = self._generate_pseudo_landmarks(
                        self.camera_states[-1]
                    )
                    pts_3d_tracked = np.concatenate([pts_3d_tracked, pseudo_3d], axis=0)
                    pts_2d_tracked = np.concatenate([pts_2d_tracked, pseudo_2d], axis=0)

                if len(pts_3d_tracked) >= 4:  # PnP needs at least 4 correspondences
                    R_pnp, t_pnp = solve_camera_pnp(
                        pts_3d=pts_3d_tracked,
                        pts_2d=pts_2d_tracked,
                        K=self.state.K,
                        dist_coeffs=self.state.k,
                        R_init=self.state.R,
                        t_init=self.state.t,
                    )
                    self.state.R = R_pnp
                    self.state.C = -R_pnp.T @ t_pnp

        if self.debug_vis.visualize:
            self.debug_vis.draw_projection(
                self.pitch_points, self.state.R, self.state.t,
                self.state.K, self.state.k,
            )
            cv2.imshow("Visualization", self.debug_vis.frame_curr)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit()

        self.camera_states.append(self.state.copy())
        self.frame_buffer.append(frame)
        return self.state

    # ========================================================================
    # PnP helpers
    # ========================================================================

    def _generate_pseudo_landmarks(
        self,
        state: CameraState,
        grid_size: int = 5,
        extent: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pseudo-landmark correspondences from a virtual ground grid.

        Projects a uniform grid of 3D points at z=0 using the *previous*
        camera pose to produce synthetic 2D correspondences that stabilise the
        PnP solver in dead zones.

        Args:
            state: Camera state from the previous frame.
            grid_size: Number of points per axis in the grid.
            extent: Half-extent (metres) of the grid around the pitch centre.

        Returns:
            pts_3d: (M, 3) virtual 3D points (z = 0).
            pts_2d: (M, 2) their 2D projections.
        """
        xs = np.linspace(-extent, extent, grid_size)
        ys = np.linspace(-extent, extent, grid_size)
        xg, yg = np.meshgrid(xs, ys)
        pts_3d = np.column_stack([xg.ravel(), yg.ravel(), np.zeros(grid_size ** 2)])

        pts_2d, _ = cv2.projectPoints(
            pts_3d, cv2.Rodrigues(state.R)[0], state.t,
            state.K, state.k,
        )
        pts_2d = pts_2d.reshape(-1, 2)

        # Keep only points that fall inside a reasonable image area
        H, W = 1080, 1920  # conservative defaults
        valid = (
            (pts_2d[:, 0] >= -100) & (pts_2d[:, 0] < W + 100) &
            (pts_2d[:, 1] >= -100) & (pts_2d[:, 1] < H + 100)
        )
        return pts_3d[valid], pts_2d[valid]

    # ========================================================================
    # Optical-flow based rotation estimation
    # ========================================================================

    def _project_pitch_points(self, K, k, R, t, img_size):
        pts_2d, _ = cv2.projectPoints(self.pitch_points, cv2.Rodrigues(R)[0], t, K, k)
        pts_2d = pts_2d.reshape(-1, 2)
        H, W = img_size[:2]
        valid = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
        return pts_2d, valid

    def _update_flow(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        state_prev: CameraState,
        state_curr: CameraState,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Update camera state from optical flow and return tracked 3D↔2D
        correspondences for PnP.

        Returns:
            pts_3d_tracked: (M, 3) world points that were successfully tracked, or None.
            pts_2d_tracked: (M, 2) their observed 2D positions in the current frame, or None.
        """
        pts_2d_prev, valid = self._project_pitch_points(
            state_prev.K, state_prev.k, state_prev.R, state_prev.t, frame.shape[:2]
        )

        pts_3d_visible = self.pitch_points[valid]
        pts_2d_prev = pts_2d_prev[valid].astype(np.float32)

        if len(pts_2d_prev) < 3:
            return None, None

        pts_2d_next, status = optical_flow_pyrlk(prev_frame, frame, pts_2d_prev)

        self.debug_vis.draw_optical_flow(pts_2d_prev, pts_2d_next, status)

        # Estimate rotation from optical flow
        pts_2d_prev_normalized = self._prep_points(pts_2d_prev[status], state_prev.K, state_prev.k)
        pts_2d_next_normalized = self._prep_points(pts_2d_next[status], state_curr.K, state_curr.k)

        M = pts_2d_next_normalized.T @ pts_2d_prev_normalized
        U, S, Vt = np.linalg.svd(M)
        R_rel = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        R = R_rel @ state_prev.R

        self.state.R = R

        # Return tracked correspondences for PnP
        pts_3d_tracked = pts_3d_visible[status]
        pts_2d_tracked = pts_2d_next[status]
        return pts_3d_tracked, pts_2d_tracked

    # ========================================================================
    # Static utility methods for coordinate transformations
    # ========================================================================

    @staticmethod
    def _prep_points(pts, K, dist):
        if dist is not None:
            dist = np.asarray(dist, dtype=np.float32).ravel()
            if dist.size == 2:  # k1,k2 only -> expand
                dist = np.array([dist[0], dist[1], 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            dist = None
        pts_ud = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=None).reshape(-1, 2)
        pts_n = np.c_[pts_ud, np.ones(pts_ud.shape[0])]
        pts_n = pts_n / np.linalg.norm(pts_n, axis=1, keepdims=True)
        return pts_n

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to yaw/pitch/roll using custom camera convention.

        This follows OpenCV camera convention where:
        - R[2] is the forward direction (camera's viewing direction) in world coords
        - R[1] is the up direction in world coords
        - R[0] is the right direction in world coords
        """
        assert R.shape == (3, 3)

        f = R[2]
        pitch = np.arcsin(np.clip(f[2], -1.0, 1.0))
        yaw = np.arctan2(f[0], f[1])

        sy, cy = np.sin(yaw), np.cos(yaw)
        r0 = np.array([cy, -sy, 0.0], dtype=np.float64)

        cr = np.dot(R[0], r0)
        sr = np.dot(R[1], r0)
        roll = np.arctan2(sr, cr)
        return yaw, pitch, roll

    @staticmethod
    def find_closest_orthogonal_matrix(A: np.ndarray) -> np.ndarray:
        """Find closest orthogonal matrix to A in terms of Frobenius norm."""
        U, _, Vt = np.linalg.svd(A)
        return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
