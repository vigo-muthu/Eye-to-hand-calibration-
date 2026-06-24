# Eye-to-hand-calibration 

This repository provides a workflow for performing **eye-to-hand calibration** between a robot manipulator and an external camera.

In an eye-to-hand setup, the camera is fixed in the workspace while the robot moves through different poses. By observing a calibration target and recording the corresponding robot poses, the transformation between the camera frame and robot base frame can be estimated.

The final output is:

```text
T_cam2base
```

which represents the camera pose with respect to the robot base frame.

---

## Calibration Workflow

1. Print a calibration board (Checkerboard or ChArUco board).
2. Move the robot to multiple poses.
3. Collect **at least 15–20 image and pose pairs**.
4. Capture images.
5. Record robot poses with respect to the images.
6. Solve for the camera-to-base transformation.
7. Save the resulting transformation matrix.

---

## Repository Structure

### Data Collection

| File           | Description                                               |
| -------------- | --------------------------------------------------------- |
| `camera.py` or `cam_aruco.py`    | Captures calibration images using a checkerboard target or ChArUco board.  |
| `hand.py`      | Records robot poses for each captured image. |


### Calibration Solver

| File             | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `solver.py`      | Computes the final eye-to-hand calibration and generates `T_cam2base.npy`. |
| `T_cam2base.npy` | Saved camera-to-base transformation matrix.                                |

### Calibration Targets (In Folder `Board`)

| File                                     | Description                                            |
| ---------------------------------------- | ------------------------------------------------------ |
| `checker_160x160_5x4_30.pdf`             | Checkerboard calibration target used with `camera.py`. |
| `charuco_200x200_5x4_30_22_DICT_4X4.pdf` | ChArUco calibration target used with `cam_aruco.py`.   |

### Generated Data

| Folder           | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `Hand_data/`     | Recorded robot poses.                                       |
| `eye_data_img/`  | Captured calibration images.                                |
| `eye_data_json/` | Detected calibration target information and image metadata. |

### Utility Scripts

| File         | Description                                    |
| ------------ | ---------------------------------------------- |
| `board.py`   | Calibration board configuration and utilities. |
| `gem_ver.py` | Version/testing utility script.                |

---

## Usage

### Checkerboard Calibration

```bash
python camera.py
python hand.py
python solver.py
```

### ChArUco Calibration

```bash
python cam_aruco.py
python hand.py
python solver.py
```

After calibration, the resulting camera-to-base transformation is saved as:

```text
T_cam2base.npy
```

This matrix can be used to transform points from the camera frame into the robot base frame.

<img width="640" height="480" alt="eye_054_annotated" src="https://github.com/user-attachments/assets/cebdec1e-221e-4668-8b4f-f104714b8d1a" />


