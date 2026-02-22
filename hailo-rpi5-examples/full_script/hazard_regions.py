#!/usr/bin/env python3

import cv2
import depthai as dai
import math
import numpy as np
import json
from scipy.integrate import quad
import haptic_motor_diff as haptic
import time

controller = haptic.MotorController([17, 23, 25], pwm_freq=200)  # BCM pins for M1–M3

# Stores for each ROI index a list of last 10 is_hazard values
hazard_history = {}
WINDOW_LEN = 7

SHOW_WINDOWS = False   # Set to False for headless mode (no GUI)
DEBUG = False

    # if d < 450 or d == float('inf'):
    #     return (0.0, 0)
    # elif d <= 1500:
    #     return (1.0, 1000)
    # elif d <= 5000:
    #     return (0.3, 500)
    # else:
    #     return (0.0, 0)

def vibrate_motors_by_region(config3):
    """
    config3: List of 3 (power, duration_ms) tuples for [left, middle, right]
    Each group controls 3 motors:
        Left  -> M1, M2, M3
        Middle -> M4, M5, M6
        Right -> M7, M8, M9
    """
    if len(config3) != 3:
        raise ValueError("Expected 3 (power, duration) tuples for left, middle, right")

    # # Expand to full 9-motor config
    # full_config = (
    #     [config3[0]] * 3 +
    #     [config3[1]] * 3 +
    #     [config3[2]] * 3
    # )
    controller.vibrate_motors(config3)

# Load parameters from 'params.config'
with open('params.json', 'r') as file:
    params = json.load(file)

# Extract values
persons_height = params["persons_height_cm"]
hfov = params["hfov"]
vfov = params["vfov"]
min_distance = params["min_distance"]
max_distance = params["max_distance"]
rl_min_distance = params["rl_min_distance"]
rl_max_distance = params["rl_max_distance"]
rl_side_distance = params["rl_side_distance"]
num_cols = params["num_distance_grid_cols"]
num_rows = params["num_distance_grid_rows"]

camera_height_in_mm = (persons_height * 0.3) * 10
min_distance_for_horizontal = rl_side_distance / math.tan(math.radians(hfov / 2))

def roi_squares_at_distance(x):
    return num_cols / (x * math.tan(math.radians(vfov / 2)))

integral_result, _ = quad(lambda x: roi_squares_at_distance(x), min_distance_for_horizontal, rl_max_distance)
average_roi_squares_rl = integral_result / (rl_max_distance - min_distance_for_horizontal)
vertical_rl_squares = round(average_roi_squares_rl)
center_rl_squares = (num_cols if num_cols % 2 == 0 else num_cols - 1) // 2
center_square = (num_cols // 2) + 1 if num_cols % 2 == 1 else 0

def dist_to_power_and_time(d):
    min_x = rl_min_distance * 1000
    max_x = rl_max_distance * 1000
    # print("min_x:", min_x)
    # print("max_x:", max_x)
    # print("d:", d)
    nomalize_boundary = 0.6
    d_normalized = (((d - min_x) / (max_x - min_x)) * nomalize_boundary)
    # print("d_normalized", d_normalized)

    if d_normalized < 0 or d_normalized > nomalize_boundary:
        return (0, 0)

    power = nomalize_boundary - d_normalized

    # print("power: ", power)
    return (power, 40)

# Create pipeline
pipeline = dai.Pipeline()

# Nodes
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
camRgb = pipeline.create(dai.node.ColorCamera)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xoutRgb.setStreamName("rgb")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# RGB
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Stereo

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(False)
stereo.setSubpixel(True)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)

# ROI sizing
roi_width = 1.0 / num_cols
roi_height = 1.0 / num_rows

def get_vertical_offset(row_index, total_rows):
    return (row_index + 0.5) / total_rows

for row in range(num_rows):
    y_center = get_vertical_offset(row, num_rows)
    y_top = max(0.0, y_center - roi_height / 2)
    y_bottom = min(1.0, y_center + roi_height / 2)
    for col in range(num_cols):
        x_left = col * roi_width
        x_right = min(1.0, (col + 1) * roi_width)
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 200
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(dai.Point2f(x_left, y_top), dai.Point2f(x_right, y_bottom))
        spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
camRgb.preview.link(xoutRgb.input)


frame_count = 0
start_time = time.time()

# Run pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    fontType = cv2.FONT_HERSHEY_TRIPLEX

    


    while True:
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame()
        rgbFrame = rgbQueue.get().getCvFrame()

        depth_downscaled = depthFrame[::4]
        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1) if np.any(depth_downscaled != 0) else 0
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        depthFrameColor = depthFrameColor[:, 10:-10]  # crop here

        spatialData = spatialCalcQueue.get().getSpatialLocations()

        left_min = float('inf')
        middle_min = float('inf')
        right_min = float('inf')

        region_config = [dist_to_power_and_time(0), dist_to_power_and_time(0), dist_to_power_and_time(0)]

        for i, depthData in enumerate(spatialData):
            roi = depthData.config.roi
            roiDepth = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            roiRgb = roi.denormalize(width=rgbFrame.shape[1], height=rgbFrame.shape[0])

            xmin, ymin = int(roiDepth.topLeft().x), int(roiDepth.topLeft().y)
            xmax, ymax = int(roiDepth.bottomRight().x), int(roiDepth.bottomRight().y)
            xmin_rgb, ymin_rgb = int(roiRgb.topLeft().x), int(roiRgb.topLeft().y)
            xmax_rgb, ymax_rgb = int(roiRgb.bottomRight().x), int(roiRgb.bottomRight().y)

            coords = depthData.spatialCoordinates
            distance = math.sqrt(coords.x**2 + coords.y**2 + coords.z**2)
            height = coords.y
            color = (0, 255, 255)

            is_hazard = False

            if height < (camera_height_in_mm * -1):
                color = (255, 0, 0)
            elif (camera_height_in_mm + height) <= (persons_height * 10) + 100:
                if rl_min_distance * 1000 <= distance <= min_distance_for_horizontal * 1000:
                    if DEBUG:
                        print("min distance for horizontal:", min_distance_for_horizontal * 1000)
                    is_hazard = True
                elif min_distance_for_horizontal * 1000 < distance <= rl_max_distance * 1000:
                    col = (i % num_cols) + 1
                    half = center_rl_squares
                    middle_cols = (
                        list(range((center_square - (center_rl_squares // 2)), (center_square + (center_rl_squares // 2)) + 1))
                    )
                    if DEBUG:
                        print(middle_cols)
                    if col in middle_cols:
                        is_hazard = True
                # if is_hazard:
                #     print("\n\n", i)
                #     print("Distance:", distance)
                #     print("Height:", height)
                #     print("Camera height in mm:", camera_height_in_mm)
                #     print("ROI hegiht:", camera_height_in_mm + height)
                #     print("Person relative height in mm:", (persons_height * 10) + 100)

             # --- Sliding window logic ---
            hist = hazard_history.setdefault(i, [])
            hist.append(is_hazard)
            if len(hist) > WINDOW_LEN:
                hist.pop(0)
            # True hazard only if all last 10 are True
            is_hazard_windowed = (len(hist) == WINDOW_LEN and all(hist))


            if is_hazard_windowed:
                color = (0, 0, 255)
            # --- Haptic feedback logic based on depth zones ---

        
            coords = depthData.spatialCoordinates
            # distance = math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)
            col = (i % num_cols) + 1
            if DEBUG and is_hazard_windowed:
                print("1 <", (center_square - (center_rl_squares // 2)), "<", (center_square + (center_rl_squares // 2)), "<", num_cols)
                print("col:", col)
                print("Center square:", center_square)
            if is_hazard_windowed:
                if 1 <= col < (center_square - (center_rl_squares // 2)):
                    left_min = min(left_min, distance)
                elif (center_square + (center_rl_squares // 2)) < col <= num_cols:
                    right_min = min(right_min, distance)
                elif (center_square - (center_rl_squares // 2)) <= col <= (center_square + (center_rl_squares // 2)):
                    middle_min = min(middle_min, distance)
            if is_hazard_windowed:
                region_config = [
                    dist_to_power_and_time(left_min),
                    dist_to_power_and_time(middle_min),
                    dist_to_power_and_time(right_min)
                ]


            # Draw rectangles on both frames
            if SHOW_WINDOWS:
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.rectangle(rgbFrame, (xmin_rgb, ymin_rgb), (xmax_rgb, ymax_rgb), color, 2)
                cv2.putText(depthFrameColor, "{:.1f}m".format(distance / 1000), (xmin + 5, ymin + 20), fontType, 0.5, color)
                cv2.putText(rgbFrame, "{:.1f}m".format((camera_height_in_mm + height)/1000), (xmin + 5, ymin + 20), fontType, 0.5, color)
        # print(region_config)
        vibrate_motors_by_region(region_config)
        if SHOW_WINDOWS:
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("rgb", rgbFrame)

            if cv2.waitKey(1) == ord('q'):
                break

        frame_count += 1
        if frame_count % 30 == 0:  # print every 30 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.2f}")

            
controller.close()
