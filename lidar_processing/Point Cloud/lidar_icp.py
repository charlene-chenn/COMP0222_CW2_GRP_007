import argparse
import csv
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
from sklearn.neighbors import NearestNeighbors

try:
    from rplidar_driver import LidarDriver
except ImportError:
    LidarDriver = None


# ==========================================
# PART 0: Setup
# ==========================================
# --- CONFIGURATION ---
PORT_NAME = ''
BAUD_RATE = 256000
MAX_RANGE_MM = 12000.0
MIN_RANGE_MM = 300.0
ICP_MAX_ITER = 10
CORRESPONDENCE_THRESH = 0.3
KEYFRAME_DIST_THRESH = 0.2
KEYFRAME_ANGLE_THRESH = 0.2
LOCAL_MAP_SIZE = 20

# --- BLIND SPOT FILTER (Disabled for full 360 degree swath) ---
BLIND_SPOT_MIN = -1.0
BLIND_SPOT_MAX = -1.0

# --- PYGAME VIEW SETTINGS ---
WINDOW_SIZE = 800
METERS_TO_PIXELS = 100.0
view_offset_x = WINDOW_SIZE // 2
view_offset_y = WINDOW_SIZE // 2

# --- Detect OS ---
os_name = platform.system()

# --- Assign Port based on OS ---
if os_name == 'Windows':
    PORT_NAME = 'COM8'
elif os_name == 'Darwin':
    PORT_NAME = '/dev/tty.SLAB_USBtoUART'
else:
    PORT_NAME = '/dev/ttyUSB0'

print(f"Detected {os_name}. Trying port: {PORT_NAME}")


@dataclass
class ICPRunConfig:
    max_range_mm: float = MAX_RANGE_MM
    min_range_mm: float = MIN_RANGE_MM
    beam_step: int = 1
    scan_step: int = 1
    voxel_size: float = 0.0
    max_scans: int | None = None
    blind_spot_min: float = BLIND_SPOT_MIN
    blind_spot_max: float = BLIND_SPOT_MAX
    mirror_start: int = 0
    mirror_end: int = 0
    mirror_dist: float = 2500.0
    progress_interval: int = 250


@dataclass
class ICPRunResult:
    trajectory: np.ndarray
    scan_indices: np.ndarray
    point_cloud_map: np.ndarray
    keyframe_scans: list[np.ndarray]
    keyframe_poses: np.ndarray
    keyframe_trajectory_indices: np.ndarray
    processed_scans: list[dict]
    icp_residuals: list[dict]
    metrics: dict


# ==========================================
# PART 1: MATH & ICP HELPERS
# ==========================================

def estimate_normals_pca(points, k=5):
    if len(points) < k + 1:
        return np.zeros((len(points), 2))

    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(points)
    _, indices_all = neigh.kneighbors(points)

    normals = np.zeros((points.shape[0], 2))

    for i in range(points.shape[0]):
        neighbor_points = points[indices_all[i]]
        centered = neighbor_points - np.mean(neighbor_points, axis=0)
        cov = np.dot(centered.T, centered) / k
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]
        if np.dot(normal, points[i]) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def solve_point_to_plane(src, dst, dst_normals):
    A = []
    b = []
    for i in range(len(src)):
        s = src[i]
        d = dst[i]
        n = dst_normals[i]
        # Activity 2: Calculate cross term
        cross_term = s[0] * n[1] - s[1] * n[0]
        # End of Activity 2
        A.append([cross_term, n[0], n[1]])
        b.append(np.dot(d - s, n))

    if not A:
        return np.identity(3)

    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    c, s = np.cos(x[0]), np.sin(x[0])
    R = np.array([[c, -s], [s, c]])
    T = np.identity(3)
    T[:2, :2] = R
    T[:2, 2] = [x[1], x[2]]
    return T


def icp_scan_to_map(src_points, map_points, map_normals, init_pose_guess):
    m = src_points.shape[1]
    src_h = np.ones((m+1, src_points.shape[0]))
    src_h[:m, :] = np.copy(src_points.T)
    current_global_pose = np.copy(init_pose_guess)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(map_points)

    for i in range(ICP_MAX_ITER):
        src_global_h = np.dot(current_global_pose, src_h)
        src_global = src_global_h[:2, :].T

        distances, indices = neigh.kneighbors(src_global, return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()

        mask = distances < CORRESPONDENCE_THRESH
        if np.sum(mask) < 10:
            break

        src_valid = src_global[mask]
        dst_valid = map_points[indices[mask]]
        normals_valid = map_normals[indices[mask]]

        T_delta = solve_point_to_plane(src_valid, dst_valid, normals_valid)
        current_global_pose = np.dot(T_delta, current_global_pose)

        if np.linalg.norm(T_delta[:2, 2]) < 0.001 and abs(np.arctan2(T_delta[1, 0], T_delta[0, 0])) < 0.001:
            break
    return current_global_pose


def compute_icp_alignment_stats(src_points, map_points, map_normals, pose):
    if len(src_points) == 0 or len(map_points) == 0:
        return {
            "mean_point_to_point_m": None,
            "median_point_to_point_m": None,
            "rmse_point_to_point_m": None,
            "mean_point_to_plane_m": None,
            "median_point_to_plane_m": None,
            "rmse_point_to_plane_m": None,
            "inlier_count": 0,
            "source_point_count": int(len(src_points)),
            "inlier_ratio": 0.0,
        }

    src_global = transform_points(src_points, pose)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(map_points)
    distances, indices = neigh.kneighbors(src_global, return_distance=True)
    distances = distances.ravel()
    indices = indices.ravel()

    mask = distances < CORRESPONDENCE_THRESH
    inlier_count = int(np.sum(mask))
    source_point_count = int(len(src_points))
    if inlier_count == 0:
        return {
            "mean_point_to_point_m": None,
            "median_point_to_point_m": None,
            "rmse_point_to_point_m": None,
            "mean_point_to_plane_m": None,
            "median_point_to_plane_m": None,
            "rmse_point_to_plane_m": None,
            "inlier_count": 0,
            "source_point_count": source_point_count,
            "inlier_ratio": 0.0,
        }

    src_valid = src_global[mask]
    dst_valid = map_points[indices[mask]]
    normals_valid = map_normals[indices[mask]]
    point_to_point = distances[mask]
    point_to_plane = np.abs(np.sum((dst_valid - src_valid) * normals_valid, axis=1))

    return {
        "mean_point_to_point_m": float(np.mean(point_to_point)),
        "median_point_to_point_m": float(np.median(point_to_point)),
        "rmse_point_to_point_m": float(np.sqrt(np.mean(point_to_point**2))),
        "mean_point_to_plane_m": float(np.mean(point_to_plane)),
        "median_point_to_plane_m": float(np.median(point_to_plane)),
        "rmse_point_to_plane_m": float(np.sqrt(np.mean(point_to_plane**2))),
        "inlier_count": inlier_count,
        "source_point_count": source_point_count,
        "inlier_ratio": float(inlier_count / source_point_count),
    }


def pose_to_xytheta(pose):
    return np.array([pose[0, 2], pose[1, 2], math.atan2(pose[1, 0], pose[0, 0])], dtype=float)


def pose_from_xytheta(x, y, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    pose = np.identity(3)
    pose[:2, :2] = [[c, -s], [s, c]]
    pose[:2, 2] = [x, y]
    return pose


def transform_points(points, pose):
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    points_h = np.ones((3, len(points)))
    points_h[:2, :] = points.T
    return (pose @ points_h)[:2, :].T


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points.copy()
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxels, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


# ==========================================
# PART 2: DATA CONVERSION (WITH FILTER)
# ==========================================

def load_replay_scans(file_path):
    file_path = Path(file_path)
    scans = []
    with file_path.open("r", encoding="utf-8") as handle:
        for scan_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            scans.append((scan_index, json.loads(line)))
    return scans


def process_scan(scan_data, config=None, scan_index=0):
    """ Converts [(qual, angle, dist)...] to Numpy XY (meters). """
    global MAX_RANGE_MM, MIN_RANGE_MM

    if config is None:
        max_range = MAX_RANGE_MM
        min_range = MIN_RANGE_MM
        beam_step = 1
        blind_min = BLIND_SPOT_MIN
        blind_max = BLIND_SPOT_MAX
        voxel_size = 0.0
    else:
        max_range = config.max_range_mm
        if config.mirror_start > 0 and config.mirror_start <= scan_index <= config.mirror_end:
            max_range = config.mirror_dist
        min_range = config.min_range_mm
        beam_step = max(1, int(config.beam_step))
        blind_min = config.blind_spot_min
        blind_max = config.blind_spot_max
        voxel_size = config.voxel_size

    raw = np.array(scan_data, dtype=float)
    if len(raw) == 0:
        return None

    distances = raw[:, 2]
    angles = raw[:, 1]

    # --- FILTERS ---
    # 1. Distance Filter
    dist_mask = (distances > min_range) & (distances < max_range)

    # 2. Angle Filter (Exclude Blind Spot)
    if blind_min < 0 and blind_max < 0:
        angle_mask = np.ones_like(dist_mask, dtype=bool)
    elif blind_min <= blind_max:
        angle_mask = (angles < blind_min) | (angles > blind_max)
    else:
        angle_mask = (angles < blind_min) & (angles > blind_max)

    mask = dist_mask & angle_mask
    filtered = raw[mask]
    if len(filtered) >= 10:
        filtered = filtered[::beam_step]
    if len(filtered) < 10:
        return None

    angles_rad = np.radians(filtered[:, 1])
    dists_m = filtered[:, 2] / 1000.0

    x = dists_m * np.cos(angles_rad)
    y = dists_m * np.sin(angles_rad)
    points = np.column_stack((x, y))
    points = voxel_downsample(points, voxel_size)

    if len(points) < 10:
        return None
    return points


# ==========================================
# PART 3: BATCH ICP PIPELINE
# ==========================================

def run_icp_on_scans(scans, config=None, progress_label=None):
    if config is None:
        config = ICPRunConfig()

    current_pose = np.identity(3)
    last_keyframe_pose = np.identity(3)

    keyframe_buffer = []
    global_map_points = []
    keyframe_scans = []
    keyframe_poses = []
    keyframe_trajectory_indices = []
    trajectory = []
    scan_indices = []
    processed_scans = []
    icp_residuals = []

    first_scan_done = False

    for raw_index, scan in scans:
        if raw_index % max(1, int(config.scan_step)) != 0:
            continue
        if config.max_scans is not None and len(trajectory) >= config.max_scans:
            break

        current_scan_xy = process_scan(scan, config=config, scan_index=raw_index)
        if current_scan_xy is None:
            continue

        if not first_scan_done:
            normals = estimate_normals_pca(current_scan_xy)
            keyframe_buffer.append((current_scan_xy, normals))
            global_map_points.append(current_scan_xy)
            keyframe_scans.append(current_scan_xy)
            keyframe_poses.append(pose_to_xytheta(current_pose))
            trajectory.append(pose_to_xytheta(current_pose))
            keyframe_trajectory_indices.append(0)
            scan_indices.append(raw_index)
            processed_scans.append({"scan_index": raw_index, "points_xy": current_scan_xy, "pose": pose_to_xytheta(current_pose)})
            icp_residuals.append({
                "scan_index": raw_index,
                "trajectory_index": 0,
                "mean_point_to_point_m": None,
                "median_point_to_point_m": None,
                "rmse_point_to_point_m": None,
                "mean_point_to_plane_m": None,
                "median_point_to_plane_m": None,
                "rmse_point_to_plane_m": None,
                "inlier_count": 0,
                "source_point_count": int(len(current_scan_xy)),
                "inlier_ratio": 0.0,
            })
            first_scan_done = True
            continue

        active_points = np.vstack([k[0] for k in keyframe_buffer])
        active_normals = np.vstack([k[1] for k in keyframe_buffer])

        current_pose = icp_scan_to_map(current_scan_xy, active_points, active_normals, current_pose)
        current_xytheta = pose_to_xytheta(current_pose)
        trajectory.append(current_xytheta)
        scan_indices.append(raw_index)
        processed_scans.append({"scan_index": raw_index, "points_xy": current_scan_xy, "pose": current_xytheta})
        alignment_stats = compute_icp_alignment_stats(current_scan_xy, active_points, active_normals, current_pose)
        alignment_stats["scan_index"] = raw_index
        alignment_stats["trajectory_index"] = len(trajectory) - 1
        icp_residuals.append(alignment_stats)

        delta_T = np.dot(np.linalg.inv(last_keyframe_pose), current_pose)
        dx, dy = delta_T[0, 2], delta_T[1, 2]
        dtheta = np.arctan2(delta_T[1, 0], delta_T[0, 0])
        dist_moved = np.sqrt(dx**2 + dy**2)

        if dist_moved > KEYFRAME_DIST_THRESH or abs(dtheta) > KEYFRAME_ANGLE_THRESH:
            curr_global = transform_points(current_scan_xy, current_pose)
            curr_normals = estimate_normals_pca(curr_global)
            keyframe_buffer.append((curr_global, curr_normals))
            global_map_points.append(curr_global)
            keyframe_scans.append(current_scan_xy)
            keyframe_poses.append(current_xytheta)
            keyframe_trajectory_indices.append(len(trajectory) - 1)

            last_keyframe_pose = np.copy(current_pose)
            if len(keyframe_buffer) > LOCAL_MAP_SIZE:
                keyframe_buffer.pop(0)

        if progress_label and config.progress_interval > 0 and len(trajectory) % config.progress_interval == 0:
            print(
                f"    {progress_label}: {len(trajectory)} scans processed, "
                f"{len(global_map_points)} keyframes"
            )

    trajectory_arr = np.vstack(trajectory) if trajectory else np.empty((0, 3), dtype=float)
    scan_indices_arr = np.asarray(scan_indices, dtype=int)
    map_points = np.vstack(global_map_points) if global_map_points else np.empty((0, 2), dtype=float)
    keyframe_poses_arr = np.vstack(keyframe_poses) if keyframe_poses else np.empty((0, 3), dtype=float)
    keyframe_trajectory_indices_arr = np.asarray(keyframe_trajectory_indices, dtype=int)

    path_len = 0.0
    if len(trajectory_arr) > 1:
        path_len = float(np.sum(np.linalg.norm(np.diff(trajectory_arr[:, :2], axis=0), axis=1)))
    closure_error = 0.0
    if len(trajectory_arr) > 1:
        closure_error = float(np.linalg.norm(trajectory_arr[-1, :2] - trajectory_arr[0, :2]))

    metrics = {
        "processed_scans": int(len(trajectory_arr)),
        "map_point_count": int(len(map_points)),
        "keyframe_count": int(len(keyframe_scans)),
        "path_length_m": path_len,
        "closure_error_m": closure_error,
        "success": bool(len(trajectory_arr) > 1 and len(map_points) > 0),
    }
    valid_residuals = [
        row for row in icp_residuals
        if row["mean_point_to_plane_m"] is not None
    ]
    if valid_residuals:
        metrics["mean_icp_point_to_plane_m"] = float(np.mean([row["mean_point_to_plane_m"] for row in valid_residuals]))
        metrics["median_icp_point_to_plane_m"] = float(np.median([row["median_point_to_plane_m"] for row in valid_residuals]))
        metrics["mean_icp_inlier_ratio"] = float(np.mean([row["inlier_ratio"] for row in valid_residuals]))
    else:
        metrics["mean_icp_point_to_plane_m"] = None
        metrics["median_icp_point_to_plane_m"] = None
        metrics["mean_icp_inlier_ratio"] = None

    return ICPRunResult(
        trajectory=trajectory_arr,
        scan_indices=scan_indices_arr,
        point_cloud_map=map_points,
        keyframe_scans=keyframe_scans,
        keyframe_poses=keyframe_poses_arr,
        keyframe_trajectory_indices=keyframe_trajectory_indices_arr,
        processed_scans=processed_scans,
        icp_residuals=icp_residuals,
        metrics=metrics,
    )


def run_icp_file(file_path, config=None):
    scans = load_replay_scans(file_path)
    return run_icp_on_scans(scans, config=config)


def save_icp_outputs(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "trajectory_icp.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scan_index", "x_m", "y_m", "theta_rad"])
        for scan_index, pose in zip(result.scan_indices, result.trajectory):
            writer.writerow([int(scan_index), pose[0], pose[1], pose[2]])

    with (output_dir / "icp_alignment_error.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scan_index",
            "trajectory_index",
            "mean_point_to_point_m",
            "median_point_to_point_m",
            "rmse_point_to_point_m",
            "mean_point_to_plane_m",
            "median_point_to_plane_m",
            "rmse_point_to_plane_m",
            "inlier_count",
            "source_point_count",
            "inlier_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.icp_residuals:
            writer.writerow(row)

    np.savez_compressed(output_dir / "point_cloud_map.npz", points_xy=result.point_cloud_map)
    with (output_dir / "run_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2)


def plot_point_cloud_map(output_path, points_xy, trajectory, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(points_xy) > 0:
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=1.0, c="black", alpha=0.6, linewidths=0)
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=1.5)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="tab:green", s=40, label="start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="tab:red", s=40, label="end")
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ==========================================
# PART 4: LIVE VIEWER
# ==========================================

def world_to_screen(point, offset_x, offset_y, scale):
    sx = int(offset_x + point[0] * scale)
    sy = int(offset_y - point[1] * scale)
    return (sx, sy)


def run_pygame_viewer(args):
    global METERS_TO_PIXELS, view_offset_x, view_offset_y, MAX_RANGE_MM, MIN_RANGE_MM

    if LidarDriver is None:
        raise ImportError("rplidar_driver.py is required for pygame replay/live viewer mode.")

    MAX_RANGE_MM = args.max_dist
    MIN_RANGE_MM = args.min_dist

    config = ICPRunConfig(
        max_range_mm=args.max_dist,
        min_range_mm=args.min_dist,
        beam_step=args.beam_step,
        scan_step=args.scan_step,
        voxel_size=args.voxel_size,
        mirror_start=args.mirror_start,
        mirror_end=args.mirror_end,
        mirror_dist=args.mirror_dist,
    )

    lidar = LidarDriver(mode='replay', filename=args.file)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("RPLIDAR SLAM (360 Deg Swath)")
    font = pygame.font.SysFont("Arial", 18)

    current_pose = np.identity(3)
    last_keyframe_pose = np.identity(3)

    keyframe_buffer = []
    global_map_points = []
    trajectory = [[0, 0]]

    first_scan_done = False
    status = "Starting"

    print("Starting SLAM... Blind spot set to [{}, {}] degrees.".format(BLIND_SPOT_MIN, BLIND_SPOT_MAX))

    try:
        iterator = lidar.iter_scans()
        scan_index = 0

        for scan in iterator:
            scan_index += 1
            if scan_index % max(1, int(args.scan_step)) != 0:
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    if event.key == pygame.K_r:
                        print("Resetting SLAM...")
                        current_pose = np.identity(3)
                        last_keyframe_pose = np.identity(3)
                        keyframe_buffer = []
                        global_map_points = []
                        trajectory = [[0, 0]]
                        first_scan_done = False
                        status = "Reset"
                        view_offset_x, view_offset_y = WINDOW_SIZE // 2, WINDOW_SIZE // 2
                    if event.key == pygame.K_SPACE:
                        view_offset_x, view_offset_y = WINDOW_SIZE // 2, WINDOW_SIZE // 2

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                view_offset_y += 5
            if keys[pygame.K_s]:
                view_offset_y -= 5
            if keys[pygame.K_a]:
                view_offset_x += 5
            if keys[pygame.K_d]:
                view_offset_x -= 5
            if keys[pygame.K_q]:
                METERS_TO_PIXELS *= 1.05
            if keys[pygame.K_e]:
                METERS_TO_PIXELS *= 0.95

            current_scan_xy = process_scan(scan, config=config, scan_index=scan_index)
            if current_scan_xy is None:
                continue

            if not first_scan_done:
                normals = estimate_normals_pca(current_scan_xy)
                keyframe_buffer.append((current_scan_xy, normals))
                global_map_points.append(current_scan_xy)
                first_scan_done = True
                status = "Initializing"
            else:
                active_points = np.vstack([k[0] for k in keyframe_buffer])
                active_normals = np.vstack([k[1] for k in keyframe_buffer])

                current_pose = icp_scan_to_map(current_scan_xy, active_points, active_normals, current_pose)
                cx, cy = current_pose[0, 2], current_pose[1, 2]
                trajectory.append([cx, cy])

                delta_T = np.dot(np.linalg.inv(last_keyframe_pose), current_pose)
                dx, dy = delta_T[0, 2], delta_T[1, 2]
                dtheta = np.arctan2(delta_T[1, 0], delta_T[0, 0])
                dist_moved = np.sqrt(dx**2 + dy**2)

                if dist_moved > KEYFRAME_DIST_THRESH or abs(dtheta) > KEYFRAME_ANGLE_THRESH:
                    status = "Keyframe Added"
                    curr_global = transform_points(current_scan_xy, current_pose)
                    curr_normals = estimate_normals_pca(curr_global)
                    keyframe_buffer.append((curr_global, curr_normals))
                    global_map_points.append(curr_global)

                    last_keyframe_pose = np.copy(current_pose)
                    if len(keyframe_buffer) > LOCAL_MAP_SIZE:
                        keyframe_buffer.pop(0)
                else:
                    status = "Tracking"

            screen.fill((128, 128, 128))

            if len(global_map_points) > 0:
                all_map_pts = np.vstack(global_map_points)
                for pt in all_map_pts[::5]:
                    px, py = world_to_screen(pt, view_offset_x, view_offset_y, METERS_TO_PIXELS)
                    if 0 <= px < WINDOW_SIZE and 0 <= py < WINDOW_SIZE:
                        screen.set_at((px, py), (0, 0, 0))

            viz_scan = transform_points(current_scan_xy, current_pose)
            for pt in viz_scan:
                px, py = world_to_screen(pt, view_offset_x, view_offset_y, METERS_TO_PIXELS)
                if 0 <= px < WINDOW_SIZE and 0 <= py < WINDOW_SIZE:
                    pygame.draw.circle(screen, (255, 0, 0), (px, py), 3)

            if len(trajectory) > 1:
                traj_pts = [world_to_screen(p, view_offset_x, view_offset_y, METERS_TO_PIXELS) for p in trajectory]
                pygame.draw.lines(screen, (0, 0, 255), False, traj_pts, 2)

            rx, ry = world_to_screen([current_pose[0, 2], current_pose[1, 2]], view_offset_x, view_offset_y, METERS_TO_PIXELS)
            pygame.draw.circle(screen, (0, 255, 0), (rx, ry), 5)

            info_text = f"{status} | Pts: {len(current_scan_xy)} | Press R to Reset"
            screen.blit(font.render(info_text, True, (255, 255, 255)), (10, 10))
            pygame.display.flip()

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        lidar.disconnect()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Run ICP SLAM on replayed Lidar data.")
    parser.add_argument('--file', type=str, default='outdoor_lidar3_scan_data.json', help='The JSON file to replay')
    parser.add_argument('--headless', action='store_true', help='Run without pygame and save outputs.')
    parser.add_argument('--output', type=str, default='icp_output', help='Output folder for headless mode.')
    parser.add_argument('--max_dist', type=float, default=12000.0, help='Maximum distance filter in mm.')
    parser.add_argument('--min_dist', type=float, default=220.0, help='Minimum distance filter in mm.')
    parser.add_argument('--beam_step', type=int, default=1, help='Keep every n-th beam.')
    parser.add_argument('--scan_step', type=int, default=1, help='Process every n-th scan.')
    parser.add_argument('--voxel_size', type=float, default=0.0, help='Voxel size in meters. 0 disables voxel filtering.')
    parser.add_argument('--max_scans', type=int, default=None, help='Optional scan cap for quick tests.')
    parser.add_argument('--mirror_start', type=int, default=0, help='Scan index to start mirror filter.')
    parser.add_argument('--mirror_end', type=int, default=0, help='Scan index to end mirror filter.')
    parser.add_argument('--mirror_dist', type=float, default=2500.0, help='Max distance filter during mirror section.')
    args = parser.parse_args()

    if args.headless:
        config = ICPRunConfig(
            max_range_mm=args.max_dist,
            min_range_mm=args.min_dist,
            beam_step=args.beam_step,
            scan_step=args.scan_step,
            voxel_size=args.voxel_size,
            max_scans=args.max_scans,
            mirror_start=args.mirror_start,
            mirror_end=args.mirror_end,
            mirror_dist=args.mirror_dist,
        )
        result = run_icp_file(args.file, config=config)
        output_dir = Path(args.output)
        save_icp_outputs(output_dir, result)
        plot_point_cloud_map(output_dir / "point_cloud_map.png", result.point_cloud_map, result.trajectory, "Point Cloud Map")
    else:
        run_pygame_viewer(args)


if __name__ == "__main__":
    main()
