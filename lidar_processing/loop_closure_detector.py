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
MIN_INDEX_SEPARATION = 80
MAX_DISTANCE_M = 1.25
MAX_HEADING_DIFF_RAD = 0.75
MAX_ALIGNMENT_ERROR_M = 0.35
LOOP_ICP_MAX_ITER = 15
LOOP_CORRESPONDENCE_THRESH = 0.45


# ==========================================
# PART 1: HELPERS
# ==========================================

def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def transform_points(points_xy, pose_xytheta):
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=float)
    x, y, theta = pose_xytheta
    c = math.cos(theta)
    s = math.sin(theta)
    gx = points_xy[:, 0] * c - points_xy[:, 1] * s + x
    gy = points_xy[:, 0] * s + points_xy[:, 1] * c + y
    return np.column_stack((gx, gy))


def pose_matrix_from_xytheta(pose_xytheta):
    x, y, theta = pose_xytheta
    c = math.cos(theta)
    s = math.sin(theta)
    pose = np.identity(3)
    pose[:2, :2] = [[c, -s], [s, c]]
    pose[:2, 2] = [x, y]
    return pose


def xytheta_from_pose_matrix(pose):
    return np.array([pose[0, 2], pose[1, 2], math.atan2(pose[1, 0], pose[0, 0])], dtype=float)


def relative_pose(a, b):
    ax, ay, at = a
    bx, by, bt = b
    dx = bx - ax
    dy = by - ay
    c = math.cos(at)
    s = math.sin(at)
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    rel_t = angle_diff(bt, at)
    return np.array([rel_x, rel_y, rel_t], dtype=float)


def estimate_normals_pca(points, k=5):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return np.zeros((len(points), 2), dtype=float)

    if len(points) < k + 1:
        return np.zeros((len(points), 2), dtype=float)

    model = NearestNeighbors(n_neighbors=k + 1)
    model.fit(points)
    _, indices_all = model.kneighbors(points)

    normals = np.zeros((points.shape[0], 2), dtype=float)
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
        cross_term = s[0] * n[1] - s[1] * n[0]
        A.append([cross_term, n[0], n[1]])
        b.append(np.dot(d - s, n))

    if not A:
        return np.identity(3)

    A = np.array(A)
    b = np.array(b)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    c, s = np.cos(x[0]), np.sin(x[0])
    T = np.identity(3)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 2] = [x[1], x[2]]
    return T


def refine_loop_measurement(current_scan, previous_scan, current_pose, previous_pose):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return None, float("inf")

    if len(current_scan) < 10 or len(previous_scan) < 10:
        return None, float("inf")

    init_relative = pose_matrix_from_xytheta(relative_pose(previous_pose, current_pose))
    current_relative = np.copy(init_relative)
    previous_normals = estimate_normals_pca(previous_scan)

    points_h = np.ones((3, len(current_scan)))
    points_h[:2, :] = current_scan.T

    model = NearestNeighbors(n_neighbors=1)
    model.fit(previous_scan)

    last_error = float("inf")
    for _ in range(LOOP_ICP_MAX_ITER):
        transformed = (current_relative @ points_h)[:2, :].T
        distances, indices = model.kneighbors(transformed)
        distances = distances.ravel()
        indices = indices.ravel()
        mask = distances < LOOP_CORRESPONDENCE_THRESH
        if np.sum(mask) < 10:
            break

        src_valid = transformed[mask]
        dst_valid = previous_scan[indices[mask]]
        normals_valid = previous_normals[indices[mask]]
        delta = solve_point_to_plane(src_valid, dst_valid, normals_valid)
        current_relative = delta @ current_relative
        last_error = float(np.median(distances[mask]))

        if np.linalg.norm(delta[:2, 2]) < 0.001 and abs(math.atan2(delta[1, 0], delta[0, 0])) < 0.001:
            break

    return xytheta_from_pose_matrix(current_relative), last_error


def alignment_error(current_scan, previous_scan, current_pose, previous_pose):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return float("inf")

    current_global = transform_points(current_scan, current_pose)
    previous_global = transform_points(previous_scan, previous_pose)
    if len(current_global) < 10 or len(previous_global) < 10:
        return float("inf")

    model = NearestNeighbors(n_neighbors=1)
    model.fit(previous_global)
    distances, _ = model.kneighbors(current_global)
    distances = distances.ravel()
    if len(distances) == 0:
        return float("inf")
    return float(np.median(distances))


# ==========================================
# PART 2: LOOP CLOSURE DETECTION
# ==========================================

def detect_loop_closures(trajectory, processed_scans):
    candidates = []
    if len(trajectory) < MIN_INDEX_SEPARATION + 2:
        return candidates

    for i in range(MIN_INDEX_SEPARATION, len(trajectory)):
        current_pose = trajectory[i]
        best_row = None
        best_score = float("inf")

        for j in range(0, i - MIN_INDEX_SEPARATION):
            previous_pose = trajectory[j]
            distance = float(np.linalg.norm(current_pose[:2] - previous_pose[:2]))
            heading = abs(angle_diff(current_pose[2], previous_pose[2]))
            score = distance + 0.25 * heading

            if score < best_score:
                best_score = score
                best_row = {
                    "from_index": i,
                    "to_index": j,
                    "distance_m": distance,
                    "heading_diff_rad": heading,
                    "score": score,
                }

        if best_row is None:
            continue

        accepted = (
            best_row["distance_m"] <= MAX_DISTANCE_M
            and best_row["heading_diff_rad"] <= MAX_HEADING_DIFF_RAD
        )

        align_error = float("inf")
        refined_relative = None
        if accepted:
            align_error = alignment_error(
                processed_scans[best_row["from_index"]]["points_xy"],
                processed_scans[best_row["to_index"]]["points_xy"],
                trajectory[best_row["from_index"]],
                trajectory[best_row["to_index"]],
            )
            refined_relative, refined_error = refine_loop_measurement(
                processed_scans[best_row["from_index"]]["points_xy"],
                processed_scans[best_row["to_index"]]["points_xy"],
                trajectory[best_row["from_index"]],
                trajectory[best_row["to_index"]],
            )
            if refined_error < align_error:
                align_error = refined_error
            accepted = refined_relative is not None and align_error <= MAX_ALIGNMENT_ERROR_M

        best_row["alignment_error_m"] = align_error
        if refined_relative is None:
            refined_relative = relative_pose(
                trajectory[best_row["to_index"]],
                trajectory[best_row["from_index"]],
            )
        best_row["relative_x_m"] = float(refined_relative[0])
        best_row["relative_y_m"] = float(refined_relative[1])
        best_row["relative_theta_rad"] = float(refined_relative[2])
        best_row["accepted"] = bool(accepted)
        candidates.append(best_row)

    return candidates


def save_loop_closure_candidates(path, candidates):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "from_index",
            "to_index",
            "distance_m",
            "heading_diff_rad",
            "alignment_error_m",
            "relative_x_m",
            "relative_y_m",
            "relative_theta_rad",
            "score",
            "accepted",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow(row)


def plot_loop_closure_scores(path, candidates):
    fig, ax = plt.subplots(figsize=(9, 4))
    if candidates:
        x = [row["from_index"] for row in candidates]
        y = [row["distance_m"] for row in candidates]
        accepted_x = [row["from_index"] for row in candidates if row["accepted"]]
        accepted_y = [row["distance_m"] for row in candidates if row["accepted"]]
        ax.plot(x, y, color="tab:blue", linewidth=1.2, label="nearest previous distance")
        if accepted_x:
            ax.scatter(accepted_x, accepted_y, c="tab:green", s=30, label="accepted")
        ax.axhline(MAX_DISTANCE_M, color="tab:red", linestyle="--", linewidth=1.0, label="distance threshold")
        ax.legend(loc="best")
    ax.set_title("Loop Closure Evidence")
    ax.set_xlabel("trajectory index")
    ax.set_ylabel("distance to candidate (m)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loop_closure_trajectory(path, trajectory, candidates):
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", linewidth=1.2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="tab:green", s=40, label="start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="tab:red", s=40, label="end")

    for row in candidates:
        if not row["accepted"]:
            continue
        i = row["from_index"]
        j = row["to_index"]
        ax.plot(
            [trajectory[i, 0], trajectory[j, 0]],
            [trajectory[i, 1], trajectory[j, 1]],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.8,
        )

    ax.set_title("Loop Closure Candidates")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_loop_closure_outputs(output_dir, trajectory, processed_scans):
    output_dir = Path(output_dir)
    candidates = detect_loop_closures(trajectory, processed_scans)
    save_loop_closure_candidates(output_dir / "loop_closure_candidates.csv", candidates)
    plot_loop_closure_scores(output_dir / "loop_closure_scores.png", candidates)
    plot_loop_closure_trajectory(output_dir / "loop_closure_trajectory.png", trajectory, candidates)
    return candidates
