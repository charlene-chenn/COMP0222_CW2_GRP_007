import argparse
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import occupancy_grid_mapping
import loop_closure_detector
import factor_graph_optimiser


# ==========================================
# PART 0: Setup
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "json"
OUTPUT_ROOT = SCRIPT_DIR / "outputs_q3"
SENSOR_MAX_RANGE_MM = 12000.0
SHORT_RANGE_MM = 6000.0


def load_lidar_icp_module():
    path = SCRIPT_DIR / "lidar_icp.py"
    spec = importlib.util.spec_from_file_location("fresh_lidar_icp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lidar_icp = load_lidar_icp_module()


# ==========================================
# PART 1: FILE HELPERS
# ==========================================

def list_datasets():
    return sorted(DATA_DIR.glob("*.json"))


def resolve_dataset(name):
    path = Path(name)
    if path.exists():
        return path.resolve()
    candidate = DATA_DIR / name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find dataset: {name}")


def write_summary_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset",
            "run_name",
            "max_range_mm",
            "beam_step",
            "voxel_size",
            "scan_step",
            "processed_scans",
            "map_point_count",
            "path_length_m",
            "closure_error_m",
            "mean_icp_point_to_plane_m",
            "median_icp_point_to_plane_m",
            "mean_icp_inlier_ratio",
            "success",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# ==========================================
# PART 2: PLOTS AND VIDEO
# ==========================================

def write_parameter_success_failure_csv(path, summary_rows):
    rows = []
    for row in summary_rows:
        rows.append({
            "run_name": row["run_name"],
            "status": "success" if row["success"] else "failure",
            "closure_error_m": row["closure_error_m"],
            "path_length_m": row["path_length_m"],
            "processed_scans": row["processed_scans"],
            "map_point_count": row["map_point_count"],
        })

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_name",
            "status",
            "closure_error_m",
            "path_length_m",
            "processed_scans",
            "map_point_count",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_parameter_success_failure(path, summary_rows, title="Parameter Success/Failure Summary"):
    labels = [row["run_name"] for row in summary_rows]
    values = [float(row["closure_error_m"]) for row in summary_rows]
    colors = ["tab:green" if row["success"] else "tab:red" for row in summary_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("closure error (m)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_mapping_video(path, map_points, trajectory, processed_scans=None, size=800):
    path = Path(path)
    if len(trajectory) == 0:
        return

    bounds_parts = [trajectory[:, :2]]
    if len(map_points) > 0:
        bounds_parts.append(map_points)
    if processed_scans:
        for scan in processed_scans[::max(1, len(processed_scans) // 200)]:
            points_xy = scan.get("points_xy")
            pose = scan.get("pose")
            if points_xy is None or pose is None or len(points_xy) == 0:
                continue
            scan_pose = lidar_icp.pose_from_xytheta(pose[0], pose[1], pose[2])
            bounds_parts.append(lidar_icp.transform_points(points_xy, scan_pose))

    min_xy = np.min(np.vstack(bounds_parts), axis=0)
    max_xy = np.max(np.vstack(bounds_parts), axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    margin = 0.1 * np.max(span)
    min_xy -= margin
    max_xy += margin
    span = np.maximum(max_xy - min_xy, 1.0)

    def to_pixel(points):
        px = ((points[:, 0] - min_xy[0]) / span[0] * (size - 1)).astype(int)
        py = ((points[:, 1] - min_xy[1]) / span[1] * (size - 1)).astype(int)
        return np.column_stack((px, size - 1 - py))

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (size, size))
    frame_count = min(len(trajectory), 400)
    step = max(1, len(trajectory) // frame_count)

    map_pixels = to_pixel(map_points) if len(map_points) > 0 else np.empty((0, 2), dtype=int)

    for end in range(1, len(trajectory) + 1, step):
        frame = np.full((size, size, 3), 245, dtype=np.uint8)
        if len(map_pixels) > 0:
            upto = int(len(map_pixels) * end / len(trajectory))
            for x, y in map_pixels[:upto:3]:
                if 0 <= x < size and 0 <= y < size:
                    frame[y, x] = (20, 20, 20)

        if processed_scans and end - 1 < len(processed_scans):
            scan = processed_scans[end - 1]
            points_xy = scan.get("points_xy")
            pose = scan.get("pose")
            if points_xy is not None and pose is not None and len(points_xy) > 0:
                scan_pose = lidar_icp.pose_from_xytheta(pose[0], pose[1], pose[2])
                current_scan = lidar_icp.transform_points(points_xy, scan_pose)
                scan_pixels = to_pixel(current_scan)
                for x, y in scan_pixels:
                    if 0 <= x < size and 0 <= y < size:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        traj_pixels = to_pixel(trajectory[:end, :2])
        for i in range(1, len(traj_pixels)):
            cv2.line(frame, tuple(traj_pixels[i - 1]), tuple(traj_pixels[i]), (255, 0, 0), 2)
        cv2.circle(frame, tuple(traj_pixels[-1]), 5, (0, 180, 0), -1)
        writer.write(frame)

    writer.release()


# ==========================================
# PART 3: RUNNER
# ==========================================

def experiment_configs(max_scans=None):
    return [
        ("baseline", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, max_scans=max_scans)),
        ("max_range_6000", lidar_icp.ICPRunConfig(max_range_mm=SHORT_RANGE_MM, max_scans=max_scans)),
        ("beam_step_2", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, beam_step=2, max_scans=max_scans)),
        ("beam_step_3", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, beam_step=3, max_scans=max_scans)),
        ("voxel_0p05", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, voxel_size=0.05, max_scans=max_scans)),
        ("voxel_0p10", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, voxel_size=0.10, max_scans=max_scans)),
        ("scan_step_2", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, scan_step=2, max_scans=max_scans)),
        ("scan_step_3", lidar_icp.ICPRunConfig(max_range_mm=SENSOR_MAX_RANGE_MM, scan_step=3, max_scans=max_scans)),
    ]


def build_summary_row(dataset_name, run_name, config, metrics):
    return {
        "dataset": dataset_name,
        "run_name": run_name,
        "max_range_mm": config.max_range_mm,
        "beam_step": config.beam_step,
        "voxel_size": config.voxel_size,
        "scan_step": config.scan_step,
        "processed_scans": metrics["processed_scans"],
        "map_point_count": metrics["map_point_count"],
        "path_length_m": metrics["path_length_m"],
        "closure_error_m": metrics["closure_error_m"],
        "mean_icp_point_to_plane_m": metrics.get("mean_icp_point_to_plane_m"),
        "median_icp_point_to_plane_m": metrics.get("median_icp_point_to_plane_m"),
        "mean_icp_inlier_ratio": metrics.get("mean_icp_inlier_ratio"),
        "success": metrics["success"],
    }


def run_dataset(dataset_path, output_root, max_scans=None, selected_run_override=None):
    dataset_name = dataset_path.stem
    dataset_output = output_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    print("")
    print(f"== Dataset: {dataset_path.name} ==")
    print("Loading scans...")
    scans = lidar_icp.load_replay_scans(dataset_path)
    print(f"Loaded {len(scans)} scans")
    run_records = []
    summary_rows = []

    for run_name, config in experiment_configs(max_scans=max_scans):
        print("")
        print(f"  [{run_name}] ICP odometry")
        run_output = dataset_output / run_name
        run_output.mkdir(parents=True, exist_ok=True)

        result = lidar_icp.run_icp_on_scans(scans, config=config, progress_label=run_name)
        print(
            f"    done: {result.metrics['processed_scans']} poses, "
            f"{result.metrics['map_point_count']} map points, "
            f"closure error {result.metrics['closure_error_m']:.3f} m"
        )

        print(f"  [{run_name}] Saving point cloud outputs")
        lidar_icp.save_icp_outputs(run_output, result)
        lidar_icp.plot_point_cloud_map(
            run_output / "point_cloud_map.png",
            result.point_cloud_map,
            result.trajectory,
            f"{dataset_name} - {run_name} - Point Cloud",
        )

        print(f"  [{run_name}] Building occupancy grid")
        occupancy_grid = occupancy_grid_mapping.build_occupancy_grid(
            result.processed_scans,
            result.trajectory,
            progress_label=run_name,
        )
        occupancy_grid_mapping.save_occupancy_outputs(
            run_output,
            occupancy_grid,
            result.trajectory,
            f"{dataset_name} - {run_name} - Occupancy Grid",
        )

        metrics = dict(result.metrics)
        metrics["config"] = {
            "max_range_mm": config.max_range_mm,
            "min_range_mm": config.min_range_mm,
            "beam_step": config.beam_step,
            "scan_step": config.scan_step,
            "voxel_size": config.voxel_size,
        }
        save_json(run_output / "run_metrics.json", metrics)

        summary_row = build_summary_row(dataset_name, run_name, config, result.metrics)
        summary_rows.append(summary_row)
        run_records.append((run_name, run_output, config, result, occupancy_grid, summary_row))

    write_parameter_success_failure_csv(dataset_output / "parameter_success_failure.csv", summary_rows)
    plot_parameter_success_failure(
        dataset_output / "parameter_success_failure.png",
        summary_rows,
        title=f"{dataset_name} Parameter Success/Failure",
    )

    if selected_run_override is not None:
        matches = [record for record in run_records if record[0] == selected_run_override]
        if not matches:
            raise ValueError(f"Selected run '{selected_run_override}' was not generated for {dataset_name}.")
        selected = matches[0]
    else:
        useful = [
            record for record in run_records
            if record[5]["success"] and float(record[5]["path_length_m"]) > 0.5
        ]
        baseline = [record for record in run_records if record[0] == "baseline"]
        candidates = useful or baseline or run_records
        selected = min(candidates, key=lambda item: float(item[5]["closure_error_m"]))
    selected_run_name, selected_output, selected_config, selected_result, selected_grid, selected_summary = selected

    print("")
    print(f"  Selected run for Q3c/Q3d: {selected_run_name}")
    print("  Detecting loop closures")
    candidates = loop_closure_detector.save_loop_closure_outputs(
        selected_output,
        selected_result.trajectory,
        selected_result.processed_scans,
    )
    accepted_count = sum(1 for row in candidates if row.get("accepted", False))
    print(f"    loop candidates: {len(candidates)}, accepted: {accepted_count}")

    print("  Saving final occupancy grid")
    occupancy_grid_mapping.plot_occupancy_grid(
        selected_output / "final_occupancy_grid.png",
        selected_grid,
        selected_result.trajectory,
        f"{dataset_name} - Final Occupancy Grid",
    )
    print("  Writing mapping video")
    make_mapping_video(
        selected_output / "mapping_video.mp4",
        selected_result.point_cloud_map,
        selected_result.trajectory,
        selected_result.processed_scans,
    )

    try:
        print("  Running factor graph optimization")
        optimized, after_map = factor_graph_optimiser.save_factor_graph_outputs(
            selected_output,
            selected_result.trajectory,
            candidates,
            selected_result.keyframe_scans,
            selected_result.keyframe_trajectory_indices,
            selected_result.point_cloud_map,
        )
        print("    factor graph done")
    except ImportError as exc:
        print(f"  Q3d skipped: {exc}")

    return summary_rows, {
        "dataset": dataset_name,
        "selected_run": selected_run_name,
        "selected_output": str(selected_output),
        "closure_error_m": selected_summary["closure_error_m"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the fresh Q3 LiDAR SLAM pipeline.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset JSON file. If omitted, all JSON logs are processed.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific dataset JSON files to run.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Output root directory.")
    parser.add_argument("--max-scans", type=int, default=None, help="Optional scan cap for smoke tests.")
    parser.add_argument("--selected-run", type=str, default=None, help="Optional run name to use for Q3c/Q3d.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = [resolve_dataset(dataset) for dataset in args.datasets]
    elif args.dataset:
        datasets = [resolve_dataset(args.dataset)]
    else:
        datasets = list_datasets()

    print("Fresh Q3 LiDAR pipeline")
    print(f"Output root: {output_root}")
    print(f"Datasets: {len(datasets)}")
    if args.max_scans is not None:
        print(f"Max scans per run: {args.max_scans}")

    all_summary_rows = []
    selected_rows = []
    for dataset_path in datasets:
        summary_rows, selected_row = run_dataset(
            dataset_path,
            output_root,
            max_scans=args.max_scans,
            selected_run_override=args.selected_run,
        )
        all_summary_rows.extend(summary_rows)
        selected_rows.append(selected_row)

    write_summary_csv(output_root / "q3_summary.csv", all_summary_rows)

    with (output_root / "selected_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "selected_run", "selected_output", "closure_error_m"])
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    print("")
    print("Done.")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
