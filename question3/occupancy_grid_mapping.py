import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# PART 0: Setup
# ==========================================
MAP_DIM = 1200
CELL_SIZE_MM = 20.0
CONFIDENCE_FREE = 0.03
CONFIDENCE_OCCUPIED = 0.20


# ==========================================
# PART 1: GRID HELPERS
# ==========================================

def pose_to_grid_xy(pose, map_dim=MAP_DIM, cell_size_mm=CELL_SIZE_MM):
    x_m, y_m = pose[0], pose[1]
    center = map_dim // 2
    gx = int(center + (x_m * 1000.0 / cell_size_mm))
    gy = int(center + (y_m * 1000.0 / cell_size_mm))
    return gx, gy


def world_to_grid_xy(x_m, y_m, map_dim=MAP_DIM, cell_size_mm=CELL_SIZE_MM):
    center = map_dim // 2
    gx = int(center + (x_m * 1000.0 / cell_size_mm))
    gy = int(center + (y_m * 1000.0 / cell_size_mm))
    return gx, gy


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


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
# PART 2: OCCUPANCY GRID MAPPING
# ==========================================

def build_occupancy_grid(processed_scans, trajectory, map_dim=MAP_DIM, cell_size_mm=CELL_SIZE_MM, progress_label=None):
    occupancy_grid = np.full((map_dim, map_dim), 0.5, dtype=np.float32)

    for scan_number, (scan_entry, pose) in enumerate(zip(processed_scans, trajectory), start=1):
        scan_xy = scan_entry["points_xy"]
        robot_x, robot_y = pose_to_grid_xy(pose, map_dim, cell_size_mm)
        if not (0 <= robot_x < map_dim and 0 <= robot_y < map_dim):
            continue

        global_points = transform_points(scan_xy, pose)
        for point in global_points:
            hit_x, hit_y = world_to_grid_xy(point[0], point[1], map_dim, cell_size_mm)
            if not (0 <= hit_x < map_dim and 0 <= hit_y < map_dim):
                continue

            line_points = bresenham_line(robot_x, robot_y, hit_x, hit_y)
            for free_x, free_y in line_points[:-1]:
                if 0 <= free_x < map_dim and 0 <= free_y < map_dim:
                    occupancy_grid[free_x, free_y] = max(0.0, occupancy_grid[free_x, free_y] - CONFIDENCE_FREE)

            occupancy_grid[hit_x, hit_y] = min(1.0, occupancy_grid[hit_x, hit_y] + CONFIDENCE_OCCUPIED)

        if progress_label and scan_number % 250 == 0:
            print(f"    {progress_label}: occupancy grid updated with {scan_number} scans")

    return occupancy_grid


def plot_occupancy_grid(output_path, occupancy_grid, trajectory, title, cell_size_mm=CELL_SIZE_MM):
    map_dim = occupancy_grid.shape[0]
    half_size_m = (map_dim * cell_size_mm) / 2000.0
    image = 1.0 - occupancy_grid.T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        image,
        origin="lower",
        cmap="gray",
        extent=[-half_size_m, half_size_m, -half_size_m, half_size_m],
        interpolation="nearest",
    )
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:cyan", linewidth=1.2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="tab:green", s=40, label="start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="tab:red", s=40, label="end")
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_occupancy_outputs(output_dir, occupancy_grid, trajectory, title):
    output_dir = Path(output_dir)
    np.save(output_dir / "occupancy_grid.npy", occupancy_grid)
    plot_occupancy_grid(output_dir / "occupancy_grid.png", occupancy_grid, trajectory, title)
