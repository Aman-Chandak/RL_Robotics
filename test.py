# fusion_tracking_demo.py
import numpy as np
from scipy.optimize import least_squares
import cv2  # for Rodrigues

# -----------------------
# utility: Rodrigues (rvec -> R) and inverse
# -----------------------
def rodrigues_to_R(r):
    R, _ = cv2.Rodrigues(r.astype(np.float64))
    return R

def R_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.flatten()

# -----------------------
# synthetic scene creation
# -----------------------
np.random.seed(1)

# Camera intrinsics
fx = fy = 400.0
cx = 320.0
cy = 240.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

# True camera pose (world -> camera)
true_rvec = np.array([0.2, -0.1, 0.05])  # axis-angle
true_t = np.array([0.5, -0.2, 1.2])      # translation
R_true = rodrigues_to_R(true_rvec)

# Map points in world coordinates (random in front of camera)
n_pts = 20
points_w = np.random.uniform([-1.0, -1.0, 2.0], [1.0, 1.0, 4.0], (n_pts, 3))

# Project to create noisy 2D measurements (pixel coords)
def project_point(R, t, X):
    Xc = R @ X + t
    u = K @ (Xc / Xc[2])
    return u[:2]

meas_2d = []
visible_idx = []
pix_noise_sigma = 1.0  # pixels
for i, X in enumerate(points_w):
    u = project_point(R_true, true_t, X)
    # keep if in image bounds (simple culling)
    if 0 <= u[0] < 640 and 0 <= u[1] < 480:
        noisy = u + np.random.randn(2) * pix_noise_sigma
        meas_2d.append(noisy)
        visible_idx.append(i)
meas_2d = np.array(meas_2d)
visible_pts = points_w[visible_idx]

# Base stations (known global positions)
BS = np.array([
    [ 1.5,  1.0, 2.5],
    [-1.0, -0.5, 3.0]
])

# Simulate ToA distance measurements (range = Euclidean distance + bias + noise)
toa_noise_sigma = 0.08  # meters, realistic mmWave-ish
bs_bias = np.array([0.02, -0.01])  # per-BS clock bias (meters)
cam_pos_global = R_true.T @ (-true_t)  # convert camera pose: world->camera => camera pos in world = -R^T * t
d_true = np.linalg.norm(cam_pos_global - BS, axis=1)
d_meas = d_true + bs_bias + np.random.randn(len(BS)) * toa_noise_sigma

# -----------------------
# parameterization: 6-vector x = [rvec(3), t(3)]
# -----------------------
def pack_pose(rvec, t):
    return np.hstack([rvec, t])

def unpack_pose(x):
    rvec = x[:3]
    t = x[3:6]
    return rvec, t

# -----------------------
# residual function combining reprojection and ToA residuals
# reproj residuals: 2 per observed point
# toa residuals: 1 per base station
# -----------------------
def residuals(x, pts_w, meas_2d, K, BS, d_meas, s_visual=1.0, s_toa=1.0, est_bias=None):
    rvec, t = unpack_pose(x)
    R = rodrigues_to_R(rvec)
    res = []

    # visual reprojection residuals
    for X, uv in zip(pts_w, meas_2d):
        Xc = R @ X + t
        if Xc[2] <= 0.05:  # behind camera or too close, give big residual to discourage
            res.extend([100.0, 100.0])
            continue
        uhat = K @ (Xc / Xc[2])
        reproj = (uhat[:2] - uv) * (1.0 / s_visual)  # scale by visual sigma if desired
        res.extend(reproj.tolist())

    # ToA residuals
    cam_pos = R.T @ (-t)  # camera position in world coordinates
    for j, L in enumerate(BS):
        pred_d = np.linalg.norm(cam_pos - L)
        bias_j = est_bias[j] if est_bias is not None else 0.0
        r_toa = (pred_d - (d_meas[j] - bias_j)) * (1.0 / s_toa)
        res.append(r_toa)

    return np.array(res)

# -----------------------
# initial guess (intentionally noisy)
# -----------------------
init_rvec = true_rvec + np.random.randn(3) * 0.3
init_t = true_t + np.random.randn(3) * 0.3
x0 = pack_pose(init_rvec, init_t)

print("True rvec:", true_rvec, "True t:", true_t)
print("Init rvec:", init_rvec, "Init t:", init_t)
print("ToA meas (m):", d_meas)

# -----------------------
# optimization: we provide reasonable scale weights (visual pixels ~1px, ToA meters ~0.08m)
# set s_visual = pixel sigma, s_toa = toa sigma to normalize residuals to ~N(0,1)
# -----------------------
s_visual = pix_noise_sigma
s_toa = toa_noise_sigma

res = least_squares(
    residuals, x0, verbose=2,
    args=(visible_pts, meas_2d, K, BS, d_meas, s_visual, s_toa, None),
    xtol=1e-10, ftol=1e-10, max_nfev=200
)

est_rvec, est_t = unpack_pose(res.x)
print("\nOptimization finished")
print("Estimated rvec:", est_rvec)
print("Estimated t:", est_t)
# compare camera positions in world coordinates
est_cam_pos = rodrigues_to_R(est_rvec).T @ (-est_t)
true_cam_pos = cam_pos_global
print("True cam pos (world):", true_cam_pos)
print("Estimated cam pos (world):", est_cam_pos)
print("Position error (m):", np.linalg.norm(est_cam_pos - true_cam_pos))
print("Rotation error (axis-angle norm):", np.linalg.norm(est_rvec - true_rvec))
