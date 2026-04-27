#!/bin/bash
# Exit on command errors but allow handling within loops
set -e

# Check if COLMAP is installed
if ! command -v colmap >/dev/null 2>&1; then
  echo "Error: COLMAP is not installed or not in your PATH."
  exit 1
fi

# Check for input arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <data_directory>"
  exit 1
fi

# Parent data directory
DATA_DIR="$1"
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Directory '$DATA_DIR' does not exist."
  exit 1
fi

# Define input and output directories
IMAGES_DIR="${DATA_DIR}/colmap_frames"
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Directory '$IMAGES_DIR' does not exist."
    echo "Error: Make sure to set up the directory in TUM format."
  exit 1
fi

echo "IMAGES_DIR=${IMAGES_DIR}"

OUT_DIR="${DATA_DIR}/colmap_output"
DATABASE="$OUT_DIR/database.db"
SPARSE_DIR="$OUT_DIR/sparse"
CAMERA_PARAMS="$OUT_DIR/camera_intrinsics.txt"
CAMERA_PARAMS_YAML="$OUT_DIR/camera_intrinsics.yaml"
PROGRESS_FILE="$OUT_DIR/.progress"
LOG_FILE="$OUT_DIR/colmap_errors.log"

mkdir -p "$OUT_DIR"
echo "" > "$LOG_FILE"  # Clear previous log file

# Load progress flags if the file exists
if [ -f "$PROGRESS_FILE" ]; then
  source "$PROGRESS_FILE"
else
  echo "# COLMAP progress tracking file" > "$PROGRESS_FILE"
fi

# Step 1: Feature Extraction
if [ "$FEATURES_DONE" != "1" ]; then
  echo "Starting feature extraction with shared intrinsics..."
  colmap feature_extractor \
    --database_path "$DATABASE" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV

  echo "FEATURES_DONE=1" >> "$PROGRESS_FILE"
else
  echo "Feature extraction already completed. Skipping."
fi

# Step 2: Feature Matching; do this sequentially. Loop
# detection enabled. Dictionary will be automaticlaly
# downloaded.
if [ "$MATCHING_DONE" != "1" ]; then
  echo "Running sequential feature matching..."
  colmap sequential_matcher --database_path "$DATABASE" \
    --SequentialMatching.loop_detection 1

  echo "MATCHING_DONE=1" >> "$PROGRESS_FILE"
else
  echo "Feature matching already completed. Skipping."
fi

# Step 3: Sparse Reconstruction
if [ "$SPARSE_DONE" != "1" ]; then
  echo "Starting sparse reconstruction..."
  mkdir -p "$SPARSE_DIR"
  colmap mapper --database_path "$DATABASE" --image_path "$IMAGES_DIR" --output_path "$SPARSE_DIR"

  echo "SPARSE_DONE=1" >> "$PROGRESS_FILE"
else
  echo "Sparse reconstruction already exists. Skipping mapping."
fi

# Step 4: Bundle Adjustment for ALL sparse reconstructions
if [ "$BUNDLE_ADJUSTMENT_DONE" != "1" ]; then
  echo "Running bundle adjustment on all sparse models..."

  for SPARSE_MODEL_DIR in "$SPARSE_DIR"/*; do
    if [ -d "$SPARSE_MODEL_DIR" ]; then
      echo "Processing: $SPARSE_MODEL_DIR"

      if ! colmap bundle_adjuster \
        --input_path "$SPARSE_MODEL_DIR" \
        --output_path "$SPARSE_MODEL_DIR" \
        --BundleAdjustment.refine_principal_point 1; then
        echo "Error: Bundle adjustment failed for $SPARSE_MODEL_DIR" | tee -a "$LOG_FILE"
      fi
    fi
  done

  echo "BUNDLE_ADJUSTMENT_DONE=1" >> "$PROGRESS_FILE"
else
  echo "Bundle adjustment already completed. Skipping."
fi

# Step 5: Export Camera Intrinsics for each valid sparse model
if [ "$EXPORT_INTRINSICS_DONE" != "1" ]; then
  for SPARSE_MODEL_DIR in "$SPARSE_DIR"/*; do
    if [ -f "$SPARSE_MODEL_DIR/cameras.bin" ]; then
      echo "Exporting camera intrinsics for $SPARSE_MODEL_DIR"
      colmap model_converter \
        --input_path "$SPARSE_MODEL_DIR" \
        --output_path "$SPARSE_MODEL_DIR" \
        --output_type TXT
    
      CAMERA_LINE=$(grep -m 1 "OPENCV" "$SPARSE_MODEL_DIR/cameras.txt")
      if [ -n "$CAMERA_LINE" ]; then
        CAMERA_FILE="$SPARSE_MODEL_DIR/camera_intrinsics.txt"
        ORBSLAM_YAML_FILE="$SPARSE_MODEL_DIR/camera_intrinsics.yaml"
        echo "$CAMERA_LINE" > "$CAMERA_FILE"

        # Extract and store camera intrinsics for OpenCV model
        CAMERA_ID=$(echo "$CAMERA_LINE" | awk '{print $1}')
        MODEL=$(echo "$CAMERA_LINE" | awk '{print $2}')
        WIDTH=$(echo "$CAMERA_LINE" | awk '{print $3}')
        HEIGHT=$(echo "$CAMERA_LINE" | awk '{print $4}')
        FX=$(echo "$CAMERA_LINE" | awk '{print $5}')
        FY=$(echo "$CAMERA_LINE" | awk '{print $6}')
        CX=$(echo "$CAMERA_LINE" | awk '{print $7}')
        CY=$(echo "$CAMERA_LINE" | awk '{print $8}')
        K1=$(echo "$CAMERA_LINE" | awk '{print $9}')
        K2=$(echo "$CAMERA_LINE" | awk '{print $10}')
        P1=$(echo "$CAMERA_LINE" | awk '{print $11}')
        P2=$(echo "$CAMERA_LINE" | awk '{print $12}')
        
        # Ensure K3 exists, otherwise set to 0.0
        K3=$(echo "$CAMERA_LINE" | awk '{print ($13 != "" ? $13 : "0.0")}')

        # Print refined camera parameters to text file
        echo -e "\nExtracted Camera Parameters for $SPARSE_MODEL_DIR:" | tee -a "$CAMERA_FILE"
        echo "Camera ID: $CAMERA_ID" | tee -a "$CAMERA_FILE"
        echo "Model: $MODEL" | tee -a "$CAMERA_FILE"
        echo "Image Size: ${WIDTH}x${HEIGHT}" | tee -a "$CAMERA_FILE"
        echo "Focal Length fx: $FX, fy: $FY" | tee -a "$CAMERA_FILE"
        echo "Principal Point cx: $CX, cy: $CY" | tee -a "$CAMERA_FILE"
        echo "Radial Distortion K1: $K1, K2: $K2, K3: $K3" | tee -a "$CAMERA_FILE"
        echo "Tangential Distortion P1: $P1, P2: $P2" | tee -a "$CAMERA_FILE"

        # Print refined camera parameters to yaml file
        echo "%YAML:1.0" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "#--------------------------------------------------------------------------------------------" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Camera Parameters. Auto-generated from COLMAP" | tee -a "$ORBSLAM_YAML_FILE"
        echo "#--------------------------------------------------------------------------------------------" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Camera calibration and distortion parameters (OpenCV)" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.fx: $FX" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.fy: $FY" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.cx: $CX" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.cy: $CY" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.k1: $K1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.k2: $K2" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.p1: $P1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.p2: $P2" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Camera frames per second" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.fps: 10.0" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Camera.RGB: 1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "#--------------------------------------------------------------------------------------------" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# ORB Parameters" | tee -a "$ORBSLAM_YAML_FILE"
        echo "#--------------------------------------------------------------------------------------------" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# ORB Extractor: Number of features per image" | tee -a "$ORBSLAM_YAML_FILE"
        echo "ORBextractor.nFeatures: 2000" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# ORB Extractor: Scale factor between levels in the scale pyramid 	" | tee -a "$ORBSLAM_YAML_FILE"
        echo "ORBextractor.scaleFactor: 1.2" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# ORB Extractor: Number of levels in the scale pyramid	" | tee -a "$ORBSLAM_YAML_FILE"
        echo "ORBextractor.nLevels: 8" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# ORB Extractor: Fast threshold" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response." | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# You can lower these values if your images have low contrast			" | tee -a "$ORBSLAM_YAML_FILE"
        echo "ORBextractor.iniThFAST: 20" | tee -a "$ORBSLAM_YAML_FILE"
        echo "ORBextractor.minThFAST: 7" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        echo "#--------------------------------------------------------------------------------------------" | tee -a "$ORBSLAM_YAML_FILE"
        echo "# Viewer Parameters" | tee -a "$ORBSLAM_YAML_FILE"
        echo "#--------------------------------------------------------------------------------------------" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.KeyFrameSize: 0.1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.KeyFrameLineWidth: 1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.GraphLineWidth: 1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.PointSize:2" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.CameraSize: 0.15" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.CameraLineWidth: 2" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.ViewpointX: 0" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.ViewpointY: -10" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.ViewpointZ: -0.1" | tee -a "$ORBSLAM_YAML_FILE"
        echo "Viewer.ViewpointF: 2000" | tee -a "$ORBSLAM_YAML_FILE"
        echo "" | tee -a "$ORBSLAM_YAML_FILE"
        


        
      else
        echo "Error: No valid OpenCV intrinsics found for $SPARSE_MODEL_DIR" | tee -a "$LOG_FILE"
      fi
    else
      echo "Error: Camera intrinsics not found for $SPARSE_MODEL_DIR" | tee -a "$LOG_FILE"
    fi
  done

  echo "EXPORT_INTRINSICS_DONE=1" >> "$PROGRESS_FILE"
else
  echo "Camera intrinsics already exported. Skipping."
fi
