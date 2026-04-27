# Camera & SLAM Pipeline

This directory contains the complete pipeline for Monocular SLAM, covering data capture, camera calibration via SfM, SLAM execution, and trajectory evaluation.

---

## 📁 Directory Structure

- **`COLMAP/`**: Project files and refined camera intrinsics derived from Structure-from-Motion.
    - Intrinsics are located at: `question2/COLMAP/<dataset>/camera_intrinsics.yaml`
- **`SLAM/`**: TUM-formatted datasets and output trajectories for evaluation.
- **`Videos/`**: Raw source recordings (.MOV).
- **`clean_frames/`**: Extracted image sequences used for both calibration and SLAM.

---

## 🚀 Script Documentation

### 1. Data Capture & Pre-processing

#### **`realsense_recorder.py`**
Captures raw video and depth data from an Intel RealSense D455.
- **Usage**: `python3 realsense_recorder.py [options]`
- **Options**:
  - `--duration <sec>`: Recording duration (default: 30).
  - `--framerate <hz>`: Camera speed (6, 15, 30, 60).
  - `--output <dir>`: Directory to save results.
  - `--no-display`: Run without the live feature-view window.
- **Output**: Generates `.png` frames, `.npy` descriptors/keypoints, and a `metadata.csv` log.

#### **`extract.py`**
Extracts frames from recorded video files.
- **Usage**: `python3 extract.py <video_path> -o <output_dir> [options]`
- **Options**:
  - `-p, --prefix`: Filename prefix (default: `frame`).
  - `-s, --start`: Starting frame index (default: `0`).

#### **`generate_tum_format.py`**
Prepares extracted frames for the ORB-SLAM2 algorithm.
- **Usage**: `python3 generate_tum_format.py <frames_dir> <output_dir> --fps <val>`
- **Output**: Creates a `rgb.txt` timestamp file and a `rgb/` symlink to the frames.

---

### 2. Calibration (COLMAP)

#### **`run_colmap.sh`**
Automates the full SfM pipeline to derive high-precision intrinsics.
- **Usage**: `./run_colmap.sh <data_directory>`
- **Workflow**:
  1. Feature Extraction (OpenCV model).
  2. Sequential Matching (with Loop Detection).
  3. Incremental Mapper.
  4. Global Bundle Adjustment (with Principal Point refinement).
- **Features**: Includes a `.progress` file to resume long runs if interrupted.

---

### 3. Execution (ORB-SLAM2)

#### **`run_orbslam.sh`**
Convenience runner for ORB-SLAM2 (Monocular TUM mode).
- **Usage**: `./run_orbslam.sh <command>`
- **Commands**:
  - `runplaygroundlong`: Automated run for the Outdoor/Playground dataset.
  - `runlobby`: Automated run for the Indoor/Lobby dataset.
  - `runclassroom`: Automated run for the Classroom dataset.
  - `run <yaml> <dataset_dir> [out]`: Fully custom execution mode.

---

### 4. Evaluation (EVO)

#### **`run_evo.py`**
Python wrapper for the `evo` package to analyze trajectory accuracy.
- **Usage**: `python3 run_evo.py <command>`
- **Commands**:
  - `runplaygroundlong`: Compares SLAM vs COLMAP for the playground dataset.
  - `runlobby`: Compares SLAM vs COLMAP for the lobby dataset.
  - `make_colmap_tum <images.txt> <rgb.txt>`: Converts COLMAP output to TUM format.
  - `run <ref.txt> <est.txt>`: Custom comparison between two trajectories.
- **Output**: Generates APE/RPE plots and `.zip` results in the dataset's `evo/` folder.

---

## 🛠 Standard Usage Workflow

1. **Extract**: `python3 extract.py Videos/lobby.MOV -o clean_frames/lobby`
2. **Format**: `python3 generate_tum_format.py clean_frames/lobby SLAM/lobby_SLAM --fps 30`
3. **Calibrate**: `./run_colmap.sh SLAM/lobby_SLAM`
4. **Run SLAM**: `./run_orbslam.sh runlobby`
5. **Evaluate**: `python3 run_evo.py runlobby`
