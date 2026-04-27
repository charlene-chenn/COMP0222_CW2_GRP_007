# Question 1: Visual SLAM with Datasets - Results Reproduction Guide

This guide explains how to reproduce all Question 1 results for COMP0222 Coursework 2.
To implement the code changes, clone this repository (https://github.com/UCL/COMP0222-249_25-26_ORB_SLAM2.git) and replace the code in Source/Libraries/ORB_SLAM2/ with the folder ORB_SLAM2 in this question1 folder.

---

## Quick Reference: The Four Tests

| Test | Task | File Changes | Command |
|------|------|--------------|---------|
| **1.a** | Baseline (default settings) | None | `./Install/bin/mono_kitti KITTI04-12.yaml <file>.txt` |
| **1.b** | Vary ORB feature count | Edit `KITTI*.yaml`: change `ORBextractor.nFeatures` | Same command as 1.a |
| **1.c** | Disable outlier rejection | Edit `Optimizer.cc` lines 703, 717 | Same command as 1.a |
| **1.d** | Disable loop closure | Edit `LoopClosing.cc` lines 60-68 | Same command as 1.a |

---
## Question 1.a: Baseline Evaluation

### Run Baseline on KITTI 07:
```bash
cd /Users/<your_directory>/COMP0222-249_25-26_ORB_SLAM2

./Install/bin/mono_kitti \
  ./Install/etc/orbslam2/Monocular/KITTI00-02.yaml \
  dataset/sequences/07 \
  test/baseline_kitti07.txt
```

### Run Baseline on TUM (your chosen long sequence):
```bash
./Install/bin/mono_tum \
  ./Install/etc/orbslam2/Monocular/TUM1.yaml \
  rgbd_dataset_freiburg2_desk_with_person \
  test/baseline_tum_fr2.txt
```

### Evaluate with EVO:
```bash
# KITTI 07
evo_ape tum dataset/poses/07.txt test/baseline_kitti07.txt \
  -as -p --plot_mode xyz --save_plot baseline_kitti07.png

# TUM
evo_ape tum rgbd_dataset_freiburg2_desk_with_person/groundtruth.txt test/baseline_tum_fr2.txt \
  -as -p --plot_mode xyz --save_plot baseline_tum_fr2.png
```

---

## Question 1.b: ORB Feature Count Reduction

### Modify Feature Count:

Edit `./Install/etc/orbslam2/Monocular/KITTI00-02.yaml`:
```yaml
# Change this line:
ORBextractor.nFeatures: 2000

# To test these values:
ORBextractor.nFeatures: 1500  # Test 1
ORBextractor.nFeatures: 1000  # Test 2
```

### Run Tests:
```bash
# Create three YAML files with different feature counts
cp ./Install/etc/orbslam2/Monocular/KITTI00-02.yaml KITTI00-02_1500.yaml
# Edit KITTI00-02_1500.yaml: ORBextractor.nFeatures: 1500

cp ./Install/etc/orbslam2/Monocular/KITTI00-02.yaml KITTI00-02_1000.yaml
# Edit KITTI00-02_1000.yaml: ORBextractor.nFeatures: 1000

# Run tests
./Install/bin/mono_kitti KITTI00-02_1500.yaml dataset/sequences/07 test/feat_1500_kitti07.txt
./Install/bin/mono_kitti KITTI00-02_1000.yaml dataset/sequences/07 test/feat_1000_kitti07.txt

# Same for TUM
./Install/bin/mono_tum KITTI00-02_1500.yaml rgbd_dataset_freiburg2_desk_with_person test/feat_1500_tum_fr2.txt
./Install/bin/mono_tum KITTI00-02_1000.yaml rgbd_dataset_freiburg2_desk_with_person test/feat_1000_tum_fr2.txt
```

### Evaluate:
```bash
evo_ape tum dataset/poses/07.txt test/feat_1500_kitti07.txt -as --t_max_diff 0.05 -p --plot_mode xyz --save_plot feat_1500_kitti07.png
evo_ape tum dataset/poses/07.txt test/feat_1000_kitti07.txt -as --t_max_diff 0.05 -p --plot_mode xyz --save_plot feat_1000_kitti07.png
```

---

## Question 1.c: Disable Outlier Rejection

### Modify Source Code:

**File:** `Source/Libraries/ORB_SLAM2/src/Optimizer.cc`

**Line 703 - Change FROM:**
```cpp
if (e->chi2() > 5.991 || !e->isDepthPositive()) {
```

**Change TO:**
```cpp
if (e->chi2() > 1e10 || !e->isDepthPositive()) {
```

**Line 717 - Change FROM:**
```cpp
if (e->chi2() > 7.815 || !e->isDepthPositive()) {
```

**Change TO:**
```cpp
if (e->chi2() > 1e10 || !e->isDepthPositive()) {
```

### Rebuild:
```bash
./Build.sh Release
```

### Run Tests:
```bash
./Install/bin/mono_kitti \
  ./Install/etc/orbslam2/Monocular/KITTI00-02.yaml \
  dataset/sequences/07 \
  test/no_outlier_rejection_kitti07.txt

./Install/bin/mono_tum \
  ./Install/etc/orbslam2/Monocular/TUM1.yaml \
  rgbd_dataset_freiburg2_desk_with_person \
  test/no_outlier_rejection_tum_fr2.txt
```

### Evaluate & Compare with Baseline:
```bash
evo_ape tum dataset/poses/07.txt test/no_outlier_rejection_kitti07.txt \
  -as --t_max_diff 0.05 -p --plot_mode xyz --save_plot no_outlier_rejection_kitti07.png

# Compare metrics
evo_ape tum dataset/poses/07.txt test/baseline_kitti07.txt test/no_outlier_rejection_kitti07.txt \
  -r full --save_results results_comparison_q1c.json
```

### Restore Original Code:
```bash
git checkout -- Source/Libraries/ORB_SLAM2/src/Optimizer.cc
./Build.sh Release  # Rebuild to restore baseline behavior
```

---

## Question 1.d: Disable Loop Closure

### Modify Source Code:

**File:** `Source/Libraries/ORB_SLAM2/src/LoopClosing.cc`

**Lines 60-68 - Change FROM:**
```cpp
if (CheckNewKeyFrames()) {
  if (DetectLoop()) {
    if (ComputeSim3()) {
      CorrectLoop();
    }
  }
}
```

**Change TO:**
```cpp
if (CheckNewKeyFrames()) {
  // Loop closure disabled for testing
  /*
  if (DetectLoop()) {
    if (ComputeSim3()) {
      CorrectLoop();
    }
  }
  */
}
```

### Rebuild:
```bash
./Build.sh Release
```

### Run Tests:
```bash
./Install/bin/mono_kitti \
  ./Install/etc/orbslam2/Monocular/KITTI00-02.yaml \
  dataset/sequences/07 \
  test/no_loop_closure_kitti07.txt

./Install/bin/mono_tum \
  ./Install/etc/orbslam2/Monocular/TUM1.yaml \
  rgbd_dataset_freiburg2_desk_with_person \
  test/no_loop_closure_tum_fr2.txt
```

### Evaluate & Compare with Baseline:
```bash
evo_ape tum dataset/poses/07.txt test/no_loop_closure_kitti07.txt \
  -as --t_max_diff 0.05 -p --plot_mode xyz --save_plot no_loop_closure_kitti07.png

# Compare metrics
evo_ape tum dataset/poses/07.txt test/baseline_kitti07.txt test/no_loop_closure_kitti07.txt \
  -r full --save_results results_comparison_q1d.json
```

### Restore Original Code:
```bash
git checkout -- Source/Libraries/ORB_SLAM2/src/LoopClosing.cc
./Build.sh Release
```

---

## Expected Results Summary

| Test | KITTI 07 | TUM fr2 |
|------|----------|---------|
| **1.a Baseline** | ~4.33 m ATE | ~0.05-0.08 m ATE |
| **1.b Reduced Features (1500)** | Higher error (~3-5 m) | Slight degradation |
| **1.b Reduced Features (1000)** | Significant error (~4-7 m) | Noticeable degradation |
| **1.c No Outlier Rejection** | Much worse (~14+ m) | Larger error spikes |
| **1.d No Loop Closure** | Significant drift (~15-50 m) | Unbounded drift |

---

## Troubleshooting

**Problem:** Results unchanged after modifying code
```bash
# Verify changes were made
grep "1e10" Source/Libraries/ORB_SLAM2/src/Optimizer.cc

# Force rebuild
rm -rf Build/
./Build.sh Release

# Check timestamp
ls -lh Install/lib/libORB_SLAM2.so
```

**Problem:** Build fails
```bash
./Build.sh Release  # Check for compilation errors
# If conda environment issue:
conda deactivate
./Build.sh Release
```

**Problem:** EVO evaluation fails
```bash
# Check file format
head test/baseline_kitti07.txt  # Should have: timestamp x y z qx qy qz qw

# Verify ground truth format
head dataset/poses/07.txt
```

---

## Files & Outputs

**Configuration Files:**
- `Install/etc/orbslam2/Monocular/KITTI00-02.yaml` (for KITTI 07)
- `Install/etc/orbslam2/Monocular/TUM1.yaml` (for TUM)

**Output Trajectories:**
- `test/baseline_kitti07.txt`
- `test/feat_1500_kitti07.txt`, `test/feat_1000_kitti07.txt`
- `test/no_outlier_rejection_kitti07.txt`
- `test/no_loop_closure_kitti07.txt`

**EVO Plots:**
- `*_kitti07.png`, `*_tum_fr2.png` (trajectory visualizations)
- `results_comparison_q1c.json`, `results_comparison_q1d.json` (detailed metrics)
