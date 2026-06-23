#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import json

# ============================================================
# CONFIGURATION (Matching image_290be1.png)
# ============================================================

SQUARES_X = 4              # "Columns" from your image
SQUARES_Y = 5              # "Rows" from your image
SQUARE_LENGTH = 0.030       # "Checker Width" 30mm -> 0.030 meters

# Calib.io scales inner markers to ~75% of the checker width by default.
# 30mm * 0.75 = 22.5mm -> 0.0225 meters. 
# (Measure one printed black marker with a ruler to verify exact size!)
MARKER_LENGTH = 0.0225     

ARUCO_DICT = cv2.aruco.DICT_4X4_50  # Matches "DICT_4X4" from your image

SAVE_DIR = "eye_data"
IMAGE_DIR = "eye_images"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# REALSENSE D415 SETUP
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

color_profile = profile.get_stream(
    rs.stream.color
).as_video_stream_profile()

intrinsics = color_profile.get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(
    intrinsics.coeffs,
    dtype=np.float64
).reshape(-1, 1)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs.flatten())

# ============================================================
# CHARUCO BOARD & DETECTOR SETUP
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

# Define the physical layout of the ChArUco board
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), 
    SQUARE_LENGTH, 
    MARKER_LENGTH, 
    dictionary
)

# CRITICAL: Tell OpenCV to look for the traditional/Calib.io board layout style
board.setLegacyPattern(True)

# Initialize the unified ChArUco detector
charuco_detector = cv2.aruco.CharucoDetector(board)

# Retrieve all predefined 3D corner positions of our virtual board layout
board_3d_corners = board.getChessboardCorners()

# ============================================================
# FIND NEXT FILE NUMBER
# ============================================================

existing = []

for f in os.listdir(SAVE_DIR):
    if f.startswith("eye_") and f.endswith(".json"):
        try:
            existing.append(
                int(f.split("_")[1].split(".")[0])
            )
        except:
            pass

sample_id = 0 if len(existing) == 0 else max(existing) + 1

print(f"\nStarting sample index: {sample_id}")

# ============================================================
# MAIN LOOP
# ============================================================

print("\n")
print("===================================")
print("s   -> save pose")
print("ESC -> exit")
print("===================================")

while True:

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    image = np.asanyarray(
        color_frame.get_data()
    )

    display = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers and interpolate ChArUco chessboard corners simultaneously
    ch_corners, ch_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    pose_valid = False

    # Check if we successfully found ChArUco chessboard corners
    if ch_ids is not None and len(ch_ids) >= 4:
        
        # Draw the high-precision intersection points on the display image
        cv2.aruco.drawDetectedCornersCharuco(display, ch_corners, ch_ids)

        # Map the detected corner IDs directly to their corresponding 3D board coordinates
        obj_points = board_3d_corners[ch_ids.flatten()]
        img_points = ch_corners

        # Calculate exact board position relative to the camera center
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            pose_valid = True

            R, _ = cv2.Rodrigues(rvec)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            x = tvec[0][0]
            y = tvec[1][0]
            z = tvec[2][0]

            cv2.putText(
                display, f"X={x:.3f} m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                display, f"Y={y:.3f} m", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                display, f"Z={z:.3f} m", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                display, f"Samples: {sample_id}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

    cv2.imshow("Eye-to-Hand Capture", display)
    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    if key == ord('s'):

        if not pose_valid:
            print(
                "\nChArUco Board not properly detected. "
                "Move board and try again."
            )
            continue

        # Save JSON
        data = {
            "sample_id": sample_id,
            "R": R.tolist(),
            "t": tvec.tolist(),
            "T": T.tolist(),
            "rvec": rvec.tolist()
        }

        json_filename = os.path.join(
            SAVE_DIR,
            f"eye_{sample_id:03d}.json"
        )

        with open(json_filename, "w") as f:
            json.dump(data, f, indent=4)

        # Save annotated image
        image_filename = os.path.join(
            IMAGE_DIR,
            f"eye_{sample_id:03d}.jpg"
        )

        cv2.imwrite(image_filename, display)

        print(f"\nSaved JSON : {json_filename}")
        print(f"Saved Image: {image_filename}")
        print(f"Translation (m): {x:.4f}, {y:.4f}, {z:.4f}")

        sample_id += 1

    # ========================================================
    # EXIT
    # ========================================================

    elif key == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
