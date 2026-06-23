#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
from pymycobot.mycobot import MyCobot
from scipy.spatial.transform import Rotation

# ============================================================
# 1. SETUP CONFIGURATION (Matches camera.py & hand.py)
# ============================================================
CHECKERBOARD = (4, 3)     # 4x3 internal corners
SQUARE_SIZE = 0.030       # 30mm squares in meters
CALIB_FILE = "T_cam2base.npy"

if not os.path.exists(CALIB_FILE):
    print(f"❌ Error: Matrix file '{CALIB_FILE}' not found in this directory!")
    sys.exit(1)

T_cam2base = np.load(CALIB_FILE)

print("🤖 Connecting to MyCobot...")
mc = MyCobot('/dev/ttyUSB0', 1000000)
time.sleep(1)

# Read the frozen robot position once
robot_coords = mc.get_coords()
if not robot_coords:
    print("❌ Error: Could not read coordinates from MyCobot. Check USB connection.")
    sys.exit(1)

# Convert robot coordinates from mm to meters
robot_x = robot_coords[0] / 1000.0
robot_y = robot_coords[1] / 1000.0
robot_z = robot_coords[2] / 1000.0

print("\n📷 Initializing RealSense D415 Camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Fetch camera calibration settings
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()
camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)

# Generate 3D target surface points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

print("\n============================================================")
print("STATIONARY CALIBRATION VERIFICATION")
print("============================================================\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
            
            # Extract Target-in-Camera transform
            _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            R_cam2board, _ = cv2.Rodrigues(rvec)
            
            T_cam2board = np.eye(4)
            T_cam2board[:3, :3] = R_cam2board
            T_cam2board[:3, 3] = tvec.flatten()
            
            # ========================================================
            # THE CALIBRATION BRIDGE MATH
            # ========================================================
            # We map the visual coordinates into the robot base system
            T_base2board = T_cam2base @ T_cam2board
            cam_x = T_base2board[0, 3]
            cam_y = T_base2board[1, 3]
            cam_z = T_base2board[2, 3]
            
            # Calculate the direct distance discrepancy in millimeters
            error_mm = np.linalg.norm(np.array([cam_x, cam_y, cam_z]) - np.array([robot_x, robot_y, robot_z])) * 1000.0
            
            # Print side-by-side live metrics
            sys.stdout.write(
                f"\rROBOT ENCODERS: X={robot_x:+.3f}m, Y={robot_y:+.3f}m, Z={robot_z:+.3f}m  ||  "
                f"CAMERA + MATRIX: X={cam_x:+.3f}m, Y={cam_y:+.3f}m, Z={cam_z:+.3f}m  ||  "
                f"Delta: {error_mm:.1f} mm"
            )
            sys.stdout.flush()
            
            color = (0, 255, 0) if error_mm < 15.0 else (0, 0, 255)
            cv2.putText(display, f"Alignment Delta: {error_mm:.1f} mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            sys.stdout.write("\r⚠️ Checkerboard blocked or out of camera view...                      ")
            sys.stdout.flush()
            cv2.putText(display, "Target Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.imshow("Stationary Verification Feed", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\n\nClosing streams...")
    pipeline.stop()
    cv2.destroyAllWindows()
