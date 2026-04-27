# Question 3: 2D LiDAR SLAM

This folder contains the code used for Question 3 of COMP0222 Coursework 02. The coursework brief asks for LiDAR odometry and mapping on three collected sequences, parameter analysis, loop closure detection, factor graph optimisation, and videos showing map construction. The report explains the experimental results; this README explains the code that generated them.

The pipeline uses the newline-delimited JSON LiDAR logs in `data/json`. The `.bag` and `.csv` utilities in `data/` are included only for data conversion and are not used by the main SLAM run once the JSON files exist.

## Folder Contents

| File | Purpose |
| --- | --- |
| `runner.py` | Main entry point. Runs the full Q3 pipeline for one, several, or all JSON datasets. |
| `lidar_icp.py` | ICP laser odometry core. Loads scans, filters beams, converts polar scans to Cartesian points, estimates motion with point-to-plane ICP, and saves point cloud maps and trajectories. |
| `occupancy_grid_mapping.py` | Builds occupancy grids from ICP poses and filtered LiDAR scans using ray tracing. |
| `loop_closure_detector.py` | Detects loop closure candidates from the ICP trajectory and validates them geometrically using scan alignment. |
| `factor_graph_optimiser.py` | Builds and optimises a SE(2) factor graph using GTSAM. |
| `rplidar_driver.py` | Optional replay/live driver used by the interactive viewer path in `lidar_icp.py`. The main runner does not require it. |
| `data/convert_csv_to_json.py` | Converts extracted scan CSV files into the JSON format used by the pipeline. |
| `data/extract_bag_to_csv.py` | Extracts scan data from ROS bag files into CSV files. |

## Requirements

Install the Python packages used by the pipeline:

```powershell
pip install numpy matplotlib scikit-learn opencv-python pygame gtsam
```

Optional conversion/viewer dependencies:

```powershell
pip install rosbags rplidar
```

`gtsam` is required for Q3(d). If it is missing, the factor graph stage is skipped with a clear error message.

## Running the Pipeline

Run from this folder:

```powershell
cd final_clone\COMP0222_CW2_GRP_007\question3
```

Run all JSON datasets in `data/json`:

```powershell
python runner.py
```

Run one dataset:

```powershell
python runner.py --dataset indoor_lidar_scan_data.json
```

Run selected datasets:

```powershell
python runner.py --datasets indoor_lidar_scan_data.json indoor_marshgate.json outdoor_lidar3_scan_data.json
```

Run a short smoke test:

```powershell
python runner.py --dataset indoor_lidar_scan_data.json --max-scans 50
```

Force the run used for loop closure, factor graph optimisation, final occupancy grid, and video:

```powershell
python runner.py --dataset indoor_lidar_scan_data.json --selected-run baseline
```

By default, outputs are written to:

```text
outputs_q3/
```

Use `--output-root` to choose a different output folder.

## Q3(b): Laser Odometry and Mapping

Q3(b) asks for laser odometry, point cloud maps, occupancy grid maps, trajectories, and analysis of parameter effects.

The implementation is centred on `lidar_icp.py`. Each raw scan is expected in the form:

```text
[quality, angle_degrees, distance_mm]
```

For each processed scan, the code:

1. filters invalid ranges using `min_range_mm` and `max_range_mm`;
2. optionally reduces angular resolution using `beam_step`;
3. converts polar measurements to local Cartesian points with `x = range*cos(angle)` and `y = range*sin(angle)`;
4. optionally applies voxel downsampling;
5. aligns the scan to a local keyframe map using point-to-plane ICP;
6. stores the estimated pose as `x, y, theta`;
7. adds a new keyframe when the robot has moved far enough in translation or rotation.

The ICP implementation follows the point-to-plane scan matching logic used in the LiDAR lab material. Surface normals are estimated with local PCA, nearest-neighbour correspondences are found with `sklearn.neighbors.NearestNeighbors`, and a least-squares point-to-plane update estimates the incremental rigid transform.

For each dataset, `runner.py` runs these Q3(b) configurations:

| Run name | Meaning |
| --- | --- |
| `baseline` | Full scan processing with the sensor maximum range. |
| `max_range_6000` | Limits maximum range to 6000 mm. |
| `beam_step_2` | Keeps every second beam. |
| `beam_step_3` | Keeps every third beam. |
| `voxel_0p05` | Applies 0.05 m voxel downsampling. |
| `voxel_0p10` | Applies 0.10 m voxel downsampling. |
| `scan_step_2` | Processes every second scan. |
| `scan_step_3` | Processes every third scan. |

For each run, the following files are saved:

```text
trajectory_icp.csv
icp_alignment_error.csv
point_cloud_map.npz
point_cloud_map.png
occupancy_grid.npy
occupancy_grid.png
run_metrics.json
```

The occupancy grid is built in `occupancy_grid_mapping.py`. It uses the ICP trajectory as the robot pose estimate, then ray-casts each LiDAR beam through the grid: cells along the ray are marked free, and the endpoint cell is marked occupied. This follows the occupancy-grid mapping approach from the labs while keeping ICP as the odometry source.

The main summary files are:

```text
outputs_q3/q3_summary.csv
outputs_q3/selected_runs.csv
```

Each dataset folder also contains:

```text
parameter_success_failure.csv
parameter_success_failure.png
```

These summarise the parameter runs for that dataset.

## ICP Alignment Error

`icp_alignment_error.csv` records scan-level residuals after ICP has aligned each scan to the local map. The most relevant values are:

```text
mean_point_to_plane_m
median_point_to_plane_m
rmse_point_to_plane_m
inlier_ratio
```

These measure local scan-to-map consistency. They do not measure ground-truth accuracy or global drift. A run can have a low ICP residual while still accumulating drift over a long loop.

`q3_summary.csv` includes aggregate ICP residual columns:

```text
mean_icp_point_to_plane_m
median_icp_point_to_plane_m
mean_icp_inlier_ratio
```

## Q3(c): Loop Closure Detection

Q3(c) asks for a loop closure mechanism, an explanation of the trigger, false-positive rejection, and evidence plots.

Loop closure is implemented in `loop_closure_detector.py`. It operates on the selected ICP run for each dataset. By default, the selected run is the successful configuration with `path_length_m > 0.5` and the lowest closure error. This can be overridden with `--selected-run`.

The detector has two stages:

1. **Pose-based candidate search**
   - recent poses are ignored using `MIN_INDEX_SEPARATION = 80`;
   - candidates must be within `MAX_DISTANCE_M = 1.25`;
   - candidates must have heading difference below `MAX_HEADING_DIFF_RAD = 0.75`.

2. **Geometric validation**
   - the current scan is compared with the earlier candidate scan;
   - a point-to-plane ICP refinement estimates the relative transform between the two scans;
   - the candidate is accepted only if the alignment error is below `MAX_ALIGNMENT_ERROR_M = 0.35`.

This is designed to avoid accepting false positives based only on spatial proximity. The accepted closures are then passed to Q3(d) as loop constraints.

For the selected run, the loop closure stage saves:

```text
loop_closure_candidates.csv
loop_closure_scores.png
loop_closure_trajectory.png
```

The score plot shows candidate distance over time and accepted closures. The trajectory plot marks accepted and rejected candidate links on the estimated path.

## Q3(d): Factor Graph Optimisation

Q3(d) asks for odometry and loop closure constraints in a factor graph, before/after trajectory and map comparisons, closure error before and after optimisation, and discussion of the final occupancy grid.

This is implemented in `factor_graph_optimiser.py` using GTSAM. Each ICP pose is represented as a `Pose2` node:

```text
x_i = (x, y, theta)
```

The graph contains:

1. a prior factor on the first pose;
2. odometry between-factors between consecutive ICP poses;
3. loop closure between-factors from accepted loop closure candidates.

Loop closure edges use the refined ICP relative transform when available. A Huber robust loss is applied to loop closure constraints to reduce the influence of imperfect matches.

For the selected run, the factor graph stage saves:

```text
trajectory_optimised.csv
trajectory_before_after.png
map_before_after.png
closure_error.csv
closure_error.png
final_occupancy_grid.png
```

`closure_error.csv` reports the Euclidean distance between the start and end pose before and after optimisation.

## Video Output

The coursework also asks for videos showing map construction. For each dataset, the selected run receives:

```text
mapping_video.mp4
```

The video is generated by `runner.py`. It draws the accumulated map, the trajectory, the current robot position, and the current scan points in red.

## Selection of the Run Used for Q3(c), Q3(d), and Video

All parameter runs are generated for Q3(b). The additional loop closure, factor graph, final occupancy grid, and video outputs are generated only for one selected run per dataset.

Unless `--selected-run` is provided, `runner.py` selects:

```text
the successful run with path_length_m > 0.5 and the lowest closure_error_m
```

If no useful run exists, it falls back to the baseline run if available.

## Data Format

The main pipeline reads JSON logs from:

```text
data/json/
```

Each line is one LiDAR scan, and each scan is a JSON list of beams:

```json
[[quality, angle_degrees, distance_mm], ...]
```

The main pipeline does not read ROS bag files or CSV files directly. If needed, use:

```powershell
python data\extract_bag_to_csv.py
python data\convert_csv_to_json.py path\to\scan_data.csv
```

## Notes

- The ICP odometry is locally accurate but can drift globally, especially in long corridors or repetitive outdoor geometry.
- The occupancy grid uses the ICP or optimised poses as input; it does not replace ICP with grid-score odometry.
- Loop closure is checked over processed scan poses, not only over keyframes.
- `rplidar_driver.py` is optional for the main runner, but useful for the interactive replay/viewer mode in `lidar_icp.py`.
