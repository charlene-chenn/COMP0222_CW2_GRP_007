# Question 1: Visual SLAM with Datasets - Results Reproduction Guide

This guide explains how to reproduce all Question 1 results for COMP0222 Coursework 2.
To implement the code changes, clone this repository (https://github.com/UCL/COMP0222-249_25-26_ORB_SLAM2.git) and replace the code in Source/Libraries/ORB_SLAM2/ with the folder ORB_SLAM2 in this question1 folder.

---

## Quick Reference

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
  ./Install/etc/orbslam2/Monocular/KITTI04-12.yaml \
  dataset/sequences/07 \
  test/baseline_kitti07.txt
```

### Run Baseline on TUM (your chosen long sequence):
```bash
./Install/bin/mono_tum \
  ./Install/etc/orbslam2/Monocular/TUM2.yaml \
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

Edit `./Install/etc/orbslam2/Monocular/KITTI04-12.yaml`:
```yaml
# Change this line:
ORBextractor.nFeatures: 2000

# To test these values:
ORBextractor.nFeatures: 1200  # Test 1
ORBextractor.nFeatures: 1500  # Test 2
ORBextractor.nFeatures: 1750  # Test 2
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
