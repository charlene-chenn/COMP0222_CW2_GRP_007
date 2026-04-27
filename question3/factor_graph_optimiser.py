import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# PART 0: Setup
# ==========================================

def require_gtsam():
    try:
        import gtsam
    except ImportError as exc:
        raise ImportError(
            "GTSAM is required for Q3d factor graph optimization. "
            "Install it before running Q3d, for example with: pip install gtsam"
        ) from exc
    return gtsam


# ==========================================
# PART 1: POSE HELPERS
# ==========================================

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def relative_pose(a, b):
    ax, ay, at = a
    bx, by, bt = b
    dx = bx - ax
    dy = by - ay
    c = math.cos(at)
    s = math.sin(at)
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    rel_t = wrap_angle(bt - at)
    return rel_x, rel_y, rel_t


def transform_points(points_xy, pose_xytheta):
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=float)
    x, y, theta = pose_xytheta
    c = math.cos(theta)
    s = math.sin(theta)
    gx = points_xy[:, 0] * c - points_xy[:, 1] * s + x
    gy = points_xy[:, 0] * s + points_xy[:, 1] * c + y
    return np.column_stack((gx, gy))


# ==========================================
# PART 2: FACTOR GRAPH OPTIMIZATION
# ==========================================

def optimize_trajectory(trajectory, loop_candidates):
    gtsam = require_gtsam()

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.15, 0.15, 0.10]))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.25, 0.25, 0.15]))
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        loop_noise,
    )

    for i, pose in enumerate(trajectory):
        key = gtsam.symbol('x', i)
        initial.insert(key, gtsam.Pose2(float(pose[0]), float(pose[1]), float(pose[2])))
        if i == 0:
            graph.add(gtsam.PriorFactorPose2(key, gtsam.Pose2(float(pose[0]), float(pose[1]), float(pose[2])), prior_noise))
        else:
            rel = relative_pose(trajectory[i - 1], pose)
            graph.add(gtsam.BetweenFactorPose2(
                gtsam.symbol('x', i - 1),
                key,
                gtsam.Pose2(float(rel[0]), float(rel[1]), float(rel[2])),
                odom_noise,
            ))

    for row in loop_candidates:
        if not row.get("accepted", False):
            continue
        i = int(row["from_index"])
        j = int(row["to_index"])
        if i >= len(trajectory) or j >= len(trajectory):
            continue
        if all(key in row for key in ["relative_x_m", "relative_y_m", "relative_theta_rad"]):
            rel = (float(row["relative_x_m"]), float(row["relative_y_m"]), float(row["relative_theta_rad"]))
        else:
            rel = relative_pose(trajectory[j], trajectory[i])
        graph.add(gtsam.BetweenFactorPose2(
            gtsam.symbol('x', j),
            gtsam.symbol('x', i),
            gtsam.Pose2(float(rel[0]), float(rel[1]), float(rel[2])),
            loop_noise,
        ))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    optimized = []
    for i in range(len(trajectory)):
        pose = result.atPose2(gtsam.symbol('x', i))
        optimized.append([pose.x(), pose.y(), pose.theta()])
    return np.asarray(optimized, dtype=float)


def rebuild_map_from_scans(keyframe_scans, keyframe_poses):
    map_points = []
    for scan_xy, pose in zip(keyframe_scans, keyframe_poses):
        map_points.append(transform_points(scan_xy, pose))
    if not map_points:
        return np.empty((0, 2), dtype=float)
    return np.vstack(map_points)


# ==========================================
# PART 3: OUTPUTS
# ==========================================

def save_optimized_trajectory(path, optimized_trajectory):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "x_m", "y_m", "theta_rad"])
        for i, pose in enumerate(optimized_trajectory):
            writer.writerow([i, pose[0], pose[1], pose[2]])


def closure_error(trajectory):
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2]))


def save_closure_error_outputs(output_dir, before, after):
    output_dir = Path(output_dir)
    rows = [
        ["before", closure_error(before)],
        ["after", closure_error(after)],
    ]
    with (output_dir / "closure_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "closure_error_m"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([row[0] for row in rows], [row[1] for row in rows], color=["tab:red", "tab:green"])
    ax.set_ylabel("closure error (m)")
    ax.set_title("Closure Error Before/After Optimization")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "closure_error.png", dpi=200)
    plt.close(fig)


def plot_trajectory_before_after(path, before, after):
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(before) > 0:
        ax.plot(before[:, 0], before[:, 1], color="tab:red", linewidth=1.2, label="before")
    if len(after) > 0:
        ax.plot(after[:, 0], after[:, 1], color="tab:green", linewidth=1.2, label="after")
    ax.set_title("Trajectory Before/After Loop Closure")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_map_before_after(path, before_map, after_map, before_traj, after_traj):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, points, traj, title in [
        (axes[0], before_map, before_traj, "Before"),
        (axes[1], after_map, after_traj, "After"),
    ]:
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], s=1.0, c="black", alpha=0.5, linewidths=0)
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_factor_graph_outputs(output_dir, trajectory, loop_candidates, keyframe_scans, keyframe_trajectory_indices, before_map):
    output_dir = Path(output_dir)
    optimized = optimize_trajectory(trajectory, loop_candidates)
    save_optimized_trajectory(output_dir / "trajectory_optimised.csv", optimized)
    plot_trajectory_before_after(output_dir / "trajectory_before_after.png", trajectory, optimized)
    save_closure_error_outputs(output_dir, trajectory, optimized)

    optimized_keyframe_poses = []
    for trajectory_index in keyframe_trajectory_indices:
        nearest = min(int(trajectory_index), len(optimized) - 1)
        optimized_keyframe_poses.append(optimized[nearest])
    optimized_keyframe_poses = np.asarray(optimized_keyframe_poses, dtype=float)

    after_map = rebuild_map_from_scans(keyframe_scans, optimized_keyframe_poses)
    plot_map_before_after(output_dir / "map_before_after.png", before_map, after_map, trajectory, optimized)
    return optimized, after_map
