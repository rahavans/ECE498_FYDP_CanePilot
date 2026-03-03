#!/usr/bin/env python3
"""
main.py  —  Pi 5 navigation assistant
All YOLO inference and speech-to-text run on the remote GPU server.
The Pi handles: OAK-D depth + RGB capture, spatial hazard detection,
haptic motor control, button input, and LLM scene description.

RGB capture uses DepthAI 3 Camera node (same as test_camera.py):
640x400 BGR from device, JPEG encoded on host for detection/LLM.
"""

import base64
import json
import math
import threading
import time

import cv2
import depthai as dai
import numpy as np
import requests
from scipy.integrate import quad

import haptic_motor_diff as haptic
from speak_text import speak_text
from buttons import (
    setup_button,
    register_single_click,
    register_double_click,
    register_hold,
    register_hold_release,
    register_triple_click,
)
from speech_to_text import (
    init as init_speech_to_text,
    start_listening,
    stop as stop_speech_to_text,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SHOW_WINDOWS = False  # set True only for debugging with a monitor
DEBUG = False
ENABLE_TRACKING = True
LOG_TIME = False

SERVER_URL = "https://canepilotaivisionserver.aalwan.net"
DETECT_ENDPOINT = f"{SERVER_URL}/detect"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# JPEG quality for host-side encoding (same as test_camera approach).
# Q=30 → typical outdoor frame ≈ 15–25 KB at 640×400
JPEG_QUALITY = 30

# Send a frame to /detect every N iterations of the main loop.
DETECT_EVERY_N = 3

RGB_W, RGB_H = 640, 400


def load_params():
    """Load params.json and return a dict of parameters."""
    with open("params.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parameters (from params.json)
# ---------------------------------------------------------------------------

_params = load_params()
persons_height = _params["persons_height_cm"]
hfov = _params["hfov"]
vfov = _params["vfov"]
rl_min_distance = _params["rl_min_distance"]
rl_max_distance = _params["rl_max_distance"]
rl_side_distance = _params["rl_side_distance"]
num_cols = _params["num_distance_grid_cols"]
num_rows = _params["num_distance_grid_rows"]
api_key = _params["OPENAI_API_KEY"]


# ---------------------------------------------------------------------------
# Shared state (written by main loop / detect worker, read by callbacks)
# ---------------------------------------------------------------------------

_global_rgb_frame: np.ndarray | None = None
_global_spatial_data: list | None = None
_global_data_lock = threading.Lock()

_latest_tracks: list = []
_latest_detections: list = []
_detect_result_lock = threading.Lock()

_pending_jpeg: bytes | None = None
_pending_jpeg_lock = threading.Lock()

_hazard_history: dict[int, list] = {}
WINDOW_LEN = 5

_warned_track_ids: set[int] = set()

_stop_detect_worker = threading.Event()
_stt_session = None


# ---------------------------------------------------------------------------
# Geometry (derived from params)
# ---------------------------------------------------------------------------

camera_height_mm = (persons_height * 0.5) * 10
min_dist_horizontal_mm = (rl_side_distance / math.tan(math.radians(hfov / 2))) * 1000

_integral, _ = quad(
    lambda x: num_cols / (x * math.tan(math.radians(vfov / 2))),
    min_dist_horizontal_mm / 1000,
    rl_max_distance,
)
_center_rl_squares = (num_cols if num_cols % 2 == 0 else num_cols - 1) // 2
_center_square = (num_cols // 2) + 1 if num_cols % 2 == 1 else 0


# ---------------------------------------------------------------------------
# Haptic
# ---------------------------------------------------------------------------

controller = haptic.MotorController([23, 17, 25], pwm_freq=200)


def _dist_to_haptic(d_mm: float) -> tuple:
    """Convert a distance (mm) to (power, duration_ms) for motor control."""
    lo = rl_min_distance * 1000
    hi = rl_max_distance * 1000
    cap = 0.8
    norm = ((d_mm - lo) / (hi - lo)) * cap
    if norm < 0 or norm > cap:
        return (0, 0)
    return (1, 40)


def vibrate_motors(config3: list):
    if len(config3) != 3:
        raise ValueError("Expected 3 (power, duration) tuples")
    controller.vibrate_motors(config3)


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------

def _distance_in_bbox(depth_frame: np.ndarray, bbox: tuple) -> float:
    """
    Return the modal depth (mm) of valid pixels inside bbox.
    Robust against background clutter via histogram binning.
    """
    x1, y1, x2, y2 = bbox
    fh, fw = depth_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x1 >= x2 or y1 >= y2:
        return float("inf")
    region = depth_frame[y1:y2, x1:x2]
    valid = region[region > 0]
    if len(valid) == 0:
        return float("inf")
    if len(valid) < 10:
        return float(np.median(valid))
    n_bins = max(1, int((valid.max() - valid.min()) / 50))
    hist, edges = np.histogram(valid, bins=n_bins)
    bi = int(np.argmax(hist))
    vals = valid[(valid >= edges[bi]) & (valid < edges[bi + 1])]
    return float(np.median(vals)) if len(vals) > 0 else (edges[bi] + edges[bi + 1]) / 2.0


def _bbox_hits_roi(bbox: tuple, roi_rect: tuple) -> bool:
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi_rect
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)


# ---------------------------------------------------------------------------
# Spatial hazard classification
# ---------------------------------------------------------------------------

def _classify_spatial_rois(spatial_data: list) -> tuple[list, list, list]:
    """
    Classify each spatial ROI into red (hazard), yellow (caution), or floor.
    Returns (red_rois, yellow_rois, floor_rois).
    Each ROI dict: {index, roi_rect, distance, height, roi}
    """
    red_rois = []
    yellow_rois = []
    floor_rois = []

    for i, depth_data in enumerate(spatial_data):
        roi = depth_data.config.roi
        roi_rgb = roi.denormalize(width=RGB_W, height=RGB_H)
        rect = (
            int(roi_rgb.topLeft().x),
            int(roi_rgb.topLeft().y),
            int(roi_rgb.bottomRight().x),
            int(roi_rgb.bottomRight().y),
        )

        coords = depth_data.spatialCoordinates
        distance = math.sqrt(coords.x**2 + coords.y**2 + coords.z**2)
        height = coords.y

        is_hazard = False
        floor_level = False

        if height < 0 and height <= -camera_height_mm:
            floor_level = True
        elif (camera_height_mm + height) <= persons_height * 10 + 100:
            if (camera_height_mm + height) >= persons_height * 10 - 100:
                is_hazard = True
            if rl_min_distance * 1000 <= distance <= min_dist_horizontal_mm:
                is_hazard = True
            elif min_dist_horizontal_mm < distance <= rl_max_distance * 1000:
                col = (i % num_cols) + 1
                half = _center_rl_squares // 2
                middle_cols = list(range(_center_square - half, _center_square + half + 1))
                if col in middle_cols:
                    is_hazard = True

        hist = _hazard_history.setdefault(i, [])
        hist.append(is_hazard)
        if len(hist) > WINDOW_LEN:
            hist.pop(0)
        confirmed_hazard = len(hist) == WINDOW_LEN and all(hist)

        roi_info = {
            "index": i,
            "roi_rect": rect,
            "distance": distance,
            "height": height,
            "roi": roi,
        }
        if confirmed_hazard:
            red_rois.append(roi_info)
        elif floor_level:
            floor_rois.append(roi_info)
        else:
            yellow_rois.append(roi_info)

    return red_rois, yellow_rois, floor_rois


# ---------------------------------------------------------------------------
# Detection worker (background thread)
# ---------------------------------------------------------------------------

def _looks_like_jpeg(b: bytes) -> bool:
    """Heuristic guard against partial/invalid JPEG frames."""
    return (
        isinstance(b, (bytes, bytearray))
        and len(b) > 4
        and b[:2] == b"\xFF\xD8"
        and b[-2:] == b"\xFF\xD9"
    )


def _detection_worker():
    """
    Pulls the latest pending JPEG from _pending_jpeg and posts it to /detect.
    Runs in a daemon thread; newest frame always wins.
    """
    global _pending_jpeg, _latest_tracks, _latest_detections

    sess = requests.Session()
    sess.headers["Connection"] = "keep-alive"

    while not _stop_detect_worker.is_set():
        jpeg = None
        with _pending_jpeg_lock:
            if _pending_jpeg is not None:
                jpeg, _pending_jpeg = _pending_jpeg, None

        if jpeg is None:
            time.sleep(0.008)
            continue

        if not _looks_like_jpeg(jpeg):
            print(
                f"[detect] dropping partial/invalid JPEG: len={len(jpeg)} "
                f"head={jpeg[:4].hex()} tail={jpeg[-4:].hex()}"
            )
            continue

        try:
            resp = sess.post(
                DETECT_ENDPOINT,
                files={"file": ("f.jpg", jpeg, "image/jpeg")},
                data={"enable_tracking": str(ENABLE_TRACKING).lower()},
                timeout=5,
            )
            if resp.status_code != 200:
                print(f"[detect] HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            data = resp.json()
            with _detect_result_lock:
                _latest_tracks = data.get("tracks", [])
                _latest_detections = data.get("detections", [])
        except requests.exceptions.Timeout:
            print("[detect] timeout — server busy?")
        except requests.exceptions.ConnectionError:
            print("[detect] connection error — retrying in 1 s...")
            time.sleep(1)
        except Exception as exc:
            print(f"[detect] error: {exc}")


def _enqueue_jpeg(jpeg_bytes: bytes):
    """Make jpeg_bytes available for the detect worker (newest frame wins)."""
    global _pending_jpeg
    with _pending_jpeg_lock:
        _pending_jpeg = jpeg_bytes


# ---------------------------------------------------------------------------
# LLM (scene description / question)
# ---------------------------------------------------------------------------

def _encode_frame_for_llm(rgb_frame: np.ndarray) -> str:
    """JPEG-encode at Q=50 and return base64 string."""
    _, buf = cv2.imencode(".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.b64encode(buf).decode("utf-8")


def describe_scene_in_detail():
    with _global_data_lock:
        frame = _global_rgb_frame
    if frame is None:
        return

    def _call():
        b64 = _encode_frame_for_llm(frame)
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are acting as replacement eyes for a blind person. "
                                "Describe the visual scene clearly and concisely. "
                                "Mention objects, people, their positions, and all hazards "
                                "from floor level to head level. Max 3 sentences."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 500,
        }
        r = requests.post(
            OPENAI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        if r.status_code == 200:
            speak_text(r.json()["choices"][0]["message"]["content"])
        else:
            print(f"[llm] describe error {r.status_code}: {r.text}")

    threading.Thread(target=_call, daemon=True).start()


def question_llm(question: str):
    with _global_data_lock:
        frame = _global_rgb_frame
    if frame is None:
        return

    def _call():
        b64 = _encode_frame_for_llm(frame)
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are replacement eyes for a blind person. "
                                "Answer user questions based on the image. Max 3 sentences."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
            ],
            "max_tokens": 500,
        }
        r = requests.post(
            OPENAI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        if r.status_code == 200:
            speak_text(r.json()["choices"][0]["message"]["content"])
        else:
            print(f"[llm] question error {r.status_code}: {r.text}")

    threading.Thread(target=_call, daemon=True).start()


# ---------------------------------------------------------------------------
# Button callbacks
# ---------------------------------------------------------------------------

def _single_click_identify():
    """
    Synchronous identify: grab latest frame, send to server, speak any
    hazard objects immediately (bypasses the async worker).
    """
    with _global_data_lock:
        frame = _global_rgb_frame
        spatial_data = _global_spatial_data
    if frame is None or spatial_data is None:
        return

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return

    try:
        resp = requests.post(
            DETECT_ENDPOINT,
            files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")},
            data={"enable_tracking": "false"},
            timeout=8,
        )
        if resp.status_code != 200:
            print(f"[single_click] HTTP {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        detections = resp.json().get("detections", [])
    except Exception as exc:
        print(f"[single_click] server error: {exc}")
        return

    red_rois, _, _ = _classify_spatial_rois(spatial_data)
    spoken = False
    for det in detections:
        bx, by, bw, bh = det["bbox"]
        bbox = (bx, by, bx + bw, by + bh)
        for hroi in red_rois:
            if _bbox_hits_roi(bbox, hroi["roi_rect"]):
                dist_m = hroi["distance"] / 1000
                speak_text(f"{det['class']} {dist_m:.1f} meters")
                spoken = True
                break
        if spoken:
            break
    if not spoken and red_rois:
        speak_text("Hazard detected but no object identified")


def single_click():
    _single_click_identify()


def double_click():
    speak_text("Please stand straight and point camera towards direction you want described")
    time.sleep(10)
    speak_text("Analyzing")
    describe_scene_in_detail()


def hold():
    global _stt_session
    speak_text("Listening")
    _stt_session = start_listening(max_duration=60)


def triple_click():
    pass


def hold_release():
    global _stt_session
    if _stt_session is None:
        return
    speak_text("Stop Listening")
    transcription = _stt_session.finish()
    _stt_session = None

    if not transcription:
        return
    transcription = transcription.lower()
    if "question" in transcription:
        q = transcription.replace("question", "", 1).strip()
        print(f"[button] Questioning LLM: {q!r}")
        question_llm(q)


# ---------------------------------------------------------------------------
# Timer (optional profiling)
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self):
        self.reset()

    def log(self, label: str = ""):
        if not LOG_TIME:
            return
        now = time.time()
        self._logs.append((label, now - self._last))
        self._last = now

    def print_and_reset(self):
        if not LOG_TIME:
            return
        parts = "  |  ".join(f"{l}: {e*1000:.1f}ms" for l, e in self._logs)
        total = (self._last - self._start) * 1000
        print(f"{parts}  |  Total: {total:.1f}ms")
        self.reset()

    def reset(self):
        self._logs = []
        self._start = self._last = time.time()


# ---------------------------------------------------------------------------
# DepthAI pipeline (DepthAI 3)
# ---------------------------------------------------------------------------

def build_pipeline():
    """
    Build DepthAI 3 pipeline: stereo depth, spatial ROIs, RGB Camera node.
    Returns (pipeline, depth_queue, spatial_queue, rgb_queue).
    """
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    slc = pipeline.create(dai.node.SpatialLocationCalculator)

    cam_rgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    camera_output = cam_rgb.requestOutput(
        (RGB_W, RGB_H),
        type=dai.ImgFrame.Type.BGR888p,
    )
    rgb_queue = camera_output.createOutputQueue(maxSize=4, blocking=True)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setCamera("left")
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setCamera("right")

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
    stereo.setLeftRightCheck(False)
    stereo.setSubpixel(True)
    slc.inputConfig.setWaitForMessage(False)

    roi_w = 1.0 / num_cols
    roi_h = 1.0 / num_rows
    for row in range(num_rows):
        yc = (row + 0.5) / num_rows
        yt = max(0.0, yc - roi_h / 2)
        yb = min(1.0, yc + roi_h / 2)
        for col in range(num_cols):
            xl = col * roi_w
            xr = min(1.0, (col + 1) * roi_w)
            cfg = dai.SpatialLocationCalculatorConfigData()
            cfg.depthThresholds.lowerThreshold = 200
            cfg.depthThresholds.upperThreshold = 10000
            cfg.roi = dai.Rect(dai.Point2f(xl, yt), dai.Point2f(xr, yb))
            slc.initialConfig.addROI(cfg)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(slc.inputDepth)
    depth_queue = slc.passthroughDepth.createOutputQueue(maxSize=4, blocking=False)
    spatial_queue = slc.out.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, depth_queue, spatial_queue, rgb_queue


# ---------------------------------------------------------------------------
# Main loop helpers (one iteration = one frame)
# ---------------------------------------------------------------------------

def _compute_haptic_config(red_rois: list) -> list:
    """Compute (left, middle, right) haptic (power, duration) from red ROIs."""
    left_mm = middle_mm = right_mm = float("inf")
    for roi in red_rois:
        col = (roi["index"] % num_cols) + 1
        d = roi["distance"]
        half = _center_rl_squares // 2
        if 1 <= col < _center_square - half:
            left_mm = min(left_mm, d)
        elif col > _center_square + half:
            right_mm = min(right_mm, d)
        else:
            middle_mm = min(middle_mm, d)
    return [
        _dist_to_haptic(left_mm),
        _dist_to_haptic(middle_mm),
        _dist_to_haptic(right_mm),
    ]


def _speak_hazard_warnings(
    red_rois: list,
    tracks: list,
    detections: list,
    depth_frame: np.ndarray,
):
    """Match server results to hazard ROIs and speak warnings (once per track_id)."""
    if not red_rois:
        return
    items = tracks if (ENABLE_TRACKING and tracks) else detections

    for item in items:
        if "track_id" in item:
            x1, y1, x2, y2 = item["bbox"]
            bbox = (x1, y1, x2, y2)
            label = item["class"]
            track_id = item["track_id"]
        else:
            bx, by, bw, bh = item["bbox"]
            bbox = (bx, by, bx + bw, by + bh)
            label = item["class"]
            track_id = None

        for hroi in red_rois:
            if not _bbox_hits_roi(bbox, hroi["roi_rect"]):
                continue
            dist_mm = _distance_in_bbox(depth_frame, bbox)
            dist_m = dist_mm / 1000.0
            if track_id is not None:
                if track_id not in _warned_track_ids:
                    speak_text(f"{label} {dist_m:.1f} meters")
                    _warned_track_ids.add(track_id)
                    print(f"Hazard: {label} [ID {track_id}] at {dist_m:.1f} m")
            else:
                speak_text(f"{label} {dist_m:.1f} meters")
                print(f"Hazard: {label} at {dist_m:.1f} m")
            break


def _draw_debug_overlay(rgb_frame: np.ndarray, red_rois: list, tracks: list, detections: list) -> bool:
    """
    Draw bounding boxes on rgb_frame for SHOW_WINDOWS. Returns True if user pressed 'q'.
    """
    if not SHOW_WINDOWS or rgb_frame is None:
        return False
    items = tracks if (ENABLE_TRACKING and tracks) else detections
    if not red_rois:
        items = []
    for item in items:
        if "track_id" in item:
            x1, y1, x2, y2 = item["bbox"]
            lbl = f"#{item['track_id']} {item['class']}"
        else:
            bx, by, bw, bh = item["bbox"]
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            lbl = item["class"]
        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
        cv2.putText(
            rgb_frame, lbl, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2,
        )
    cv2.imshow("rgb", rgb_frame)
    return cv2.waitKey(1) == ord("q")


def run_main_loop(pipeline, depth_queue, spatial_queue, rgb_queue):
    """Run the main capture → classify → warn → haptic loop until quit."""
    global _global_rgb_frame, _global_spatial_data

    timer = _Timer()
    frame_count = 0
    fps_start = time.time()

    pipeline.start()
    try:
        timer.log("setup")
        while True:
            # Capture
            depth_frame = depth_queue.get().getFrame()
            spatial_data = spatial_queue.get().getSpatialLocations()
            rgb_frame = rgb_queue.get().getCvFrame()
            ok, buf = cv2.imencode(
                ".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            jpeg_bytes = buf.tobytes() if ok else b""

            with _global_data_lock:
                _global_rgb_frame = rgb_frame
                _global_spatial_data = spatial_data

            timer.log("capture")

            if frame_count % DETECT_EVERY_N == 0 and jpeg_bytes and _looks_like_jpeg(jpeg_bytes):
                _enqueue_jpeg(jpeg_bytes)

            with _detect_result_lock:
                tracks = list(_latest_tracks)
                detections = list(_latest_detections)
            timer.log("server results")

            red_rois, yellow_rois, floor_rois = _classify_spatial_rois(spatial_data)
            region_config = _compute_haptic_config(red_rois)
            timer.log("spatial")

            _speak_hazard_warnings(red_rois, tracks, detections, depth_frame)
            timer.log("warn")

            if _draw_debug_overlay(rgb_frame, red_rois, tracks, detections):
                break

            vibrate_motors(region_config)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                print(f"FPS: {frame_count / elapsed:.2f}")
            timer.print_and_reset()
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# Entry: setup → run → teardown
# ---------------------------------------------------------------------------

def main():
    setup_button()
    register_single_click(single_click)
    register_double_click(double_click)
    register_hold(hold)
    register_hold_release(hold_release)
    register_triple_click(triple_click)

    init_speech_to_text(variant="base")

    detect_thread = threading.Thread(
        target=_detection_worker,
        daemon=True,
        name="detect-worker",
    )
    detect_thread.start()

    vibrate_motors([(1, 2000), (1, 2000), (1, 2000)])
    print("PROGRAM STARTED")

    pipeline, depth_queue, spatial_queue, rgb_queue = build_pipeline()
    run_main_loop(pipeline, depth_queue, spatial_queue, rgb_queue)

    _stop_detect_worker.set()
    controller.close()
    stop_speech_to_text()


if __name__ == "__main__":
    main()
