#!/usr/bin/env python3

import cv2
import depthai as dai
import math
import numpy as np
import json
from scipy.integrate import quad
import haptic_motor_diff as haptic
import time
import hailo_platform as hpf
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import base64
import requests
import threading
from speak_text import speak_text#, speak_text_non_priority, stop_non_priority_speech, stop_priority_speech
from buttons import setup_button, register_single_click, register_double_click, register_hold, register_hold_release, register_triple_click
from speech_to_text import init as init_speech_to_text, start_listening, stop as stop_speech_to_text
import sounddevice as sd
import soundfile as sf
import librosa
from transformers import WhisperProcessor

SHOW_WINDOWS = False   # Set to False for headless mode (no GUI)
DEBUG = False
enable_tracking = True
LOG_TIME = False

# Load parameters from 'params.json'
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
api_key = params["OPENAI_API_KEY"]

# Global variables
global_rgb_frame = None
global_spatial_data = None

# LOCKS
hailo_activation_lock = threading.Lock()
global_data_lock = threading.Lock()

def describe_scene_in_detail():
    rgb_frame = None
    with global_data_lock:
        global global_rgb_frame
        rgb_frame = global_rgb_frame
    def call_llm_to_describe_scene():
        """
        Captures a single RGB frame and sends it to GPT-4 Vision via OpenAI's HTTP API.
        Returns a scene description tailored for blind users.
        """
        # Encode image to base64 JPEG
        _, buffer = cv2.imencode('.jpg', rgb_frame)
        b64_image = base64.b64encode(buffer).decode('utf-8')

        # Define prompt
        prompt = (
            "You are acting as replacement eyes for a person who is blind person. Describe the visual scene in the image as clearly and concisely as possible."
            "Mention objects, people, their relative positions, any potential hazards, and overall setting."
            "Describe and include all hazards it detects in the passage from floor level all the way up to head level."
            "Make it short though as they are on the move, max 3 sentences."
        )

        # Prepare HTTP request
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            # stop_non_priority_speech()
            # stop_priority_speech()
            speak_text(response.json()["choices"][0]["message"]["content"])
        else:
            raise RuntimeError(f"Failed to get description: {response.status_code} {response.text}")
    
    threading.Thread(target=call_llm_to_describe_scene, args=(), daemon=True).start()

def question_llm(question):
    rgb_frame = None
    with global_data_lock:
        global global_rgb_frame
        rgb_frame = global_rgb_frame
    def call_llm_to_question():
        """
        Captures a single RGB frame and sends it to GPT-4 Vision via OpenAI's HTTP API.
        Returns a scene description tailored for blind users.
        """
        # Encode image to base64 JPEG
        _, buffer = cv2.imencode('.jpg', rgb_frame)
        b64_image = base64.b64encode(buffer).decode('utf-8')

        # Define prompt
        prompt = (
            "You are acting as replacement eyes for a person who is blind person."
            "Answer the user's query based on the picture given"
            "Make it short though as they are on the move, max 3 sentences."
        )

        # Prepare HTTP request
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            # stop_non_priority_speech()
            # stop_priority_speech()
            speak_text(response.json()["choices"][0]["message"]["content"])
        else:
            raise RuntimeError(f"Failed to get description: {response.status_code} {response.text}")
    
    threading.Thread(target=call_llm_to_question, args=(), daemon=True).start()

class TimerLogger:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.logs = []  # List of (description, elapsed_since_last)

    def log(self, description=""):
        if not LOG_TIME:
            return
        now = time.time()
        elapsed = now - self.last_time
        self.logs.append((description, elapsed))
        self.last_time = now
        # self.print_log_line()

    def print_log_line(self):
        if not LOG_TIME:
            return
        line = ""
        total_time = self.last_time - self.start_time
        for desc, elapsed in self.logs:
            label = desc if desc else "Unnamed"
            line += f"{label}: {elapsed:.4f}s | "
        line += f"Total: {total_time:.4f}s"
        print(line, "\n\n\n")

    def reset(self):
        self.logs = []
        self.start_time = time.time()
        self.last_time = self.start_time

    def reset_all(self):
        self.reset()

# === COCO class labels (used in YOLOv8 by default) ===
# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
#     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
#     "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
#     "toothbrush"
# ]

# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "cat", "dog", 
#     "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "skis", "snowboard", 
#     "skateboard", "surfboard", "chair", "couch", "potted plant", "bed", "dining table", 
#     "toilet", "tv", "oven", "sink", "refrigerator"
# ]

COCO_CLASS_ID_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 15: "cat", 16: "dog",
    17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra",
    23: "giraffe", 30: "skis", 31: "snowboard", 36: "skateboard", 37: "surfboard",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 74: "oven", 81: "sink", 72: "refrigerator"
}


controller = haptic.MotorController([23,17,25], pwm_freq=200)  # BCM pins for M1–M3

setup_button()
# Stores for each ROI index a list of last 10 is_hazard values
hazard_history = {}
WINDOW_LEN = 5

# Tracking for printed warnings
warned_objects = {}  # For tracking mode - stores track_ids that have been warned about
last_detection_frame = {}  # For non-tracking mode - stores last frame when object was detected
feature_extractor = "mobilenet"

# Hailo-8 setup
HEF_PATH = "resources/yolov8m.hef"
REID_HEF_PATH = "resources/mobilenetv3_large_feature_vector_075_224.hef"

timer = TimerLogger()

def preprocess_mel(audio, sr=16000):
    import librosa
    import numpy as np

    # Convert to float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    # Pad or trim to 30 seconds
    audio = audio[:480000]
    if len(audio) < 480000:
        audio = np.pad(audio, (0, 480000 - len(audio)))

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=8000
    )

    # Log scale
    mel_spec = np.log10(np.maximum(mel_spec, 1e-10))

    # Normalize to range [0, 255] for uint8
    mel_spec -= mel_spec.min()
    max_val = mel_spec.max()
    if max_val > 0:
        mel_spec /= max_val
    else:
        mel_spec = np.zeros_like(mel_spec)

    mel_spec *= 255.0

    # Resize to expected shape (500, 80)
    if mel_spec.shape[1] > 500:
        mel_spec = mel_spec[:, :500]
    elif mel_spec.shape[1] < 500:
        mel_spec = np.pad(mel_spec, ((0,0), (0, 500 - mel_spec.shape[1])), mode='constant')

    mel_spec = np.nan_to_num(mel_spec)
    mel_spec = np.expand_dims(mel_spec.T, axis=0).astype(np.uint8)

    return mel_spec

def vibrate_motors_by_region(config3):
    """
    config3: List of 3 (power, duration_ms) tuples for [left, middle, right]
    Each group controls 3 motors:
        Left  -> M1, M2, M3
        Middle -> M4, M5, M6
        Right -> M7, M8, M9
    """
    if len(config3) != 3:
        raise ValueError("Expected 3 (power, duration) tuples for left, middle and right")
    controller.vibrate_motors(config3)
    # print(config3)
    # left = config3[0][0] > 0.5
    # front = config3[1][0] > 0.5
    # right = config3[2][0] > 0.5
    # left_front = left and front
    # front_right = front and right
    # left_right = left and right
    # all_motors = left and front and right
    
    # if all_motors:
    #     speak_text_non_priority("all")
    # elif left_front:
    #     speak_text_non_priority("front left")
    # elif front_right:
    #     speak_text_non_priority("front right")
    # elif left_right:
    #     speak_text_non_priority("left right")
    # elif left:
    #     speak_text_non_priority("left")
    # elif right:
    #     speak_text_non_priority("right")
    # elif front:
    #     speak_text_non_priority("front")
    # else:
    #     speak_text_non_priority("none")
    #     stop_non_priority_speech()




def bbox_intersects_roi(bbox, roi_rect):
    """Check if bounding box intersects with ROI rectangle"""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi_rect
    
    # Check for intersection
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

def get_distance_in_bbox(depth_frame, bbox, bin_size=50):
    """
    Get the most common (mode) distance of non-zero pixels within bounding box.
    
    Args:
        depth_frame: Depth frame from camera
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        bin_size: Size of bins for grouping similar distance values (in mm)
    
    Returns:
        Most common distance value in the bounding box, or float('inf') if no valid depths
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within frame bounds
    h, w = depth_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return float('inf')
    
    roi_depth = depth_frame[y1:y2, x1:x2]
    valid_depths = roi_depth[roi_depth > 0]
    
    if len(valid_depths) == 0:
        return float('inf')
    
    # If we have very few points, just return the median
    if len(valid_depths) < 10:
        return np.median(valid_depths)
    
    # Bin the depth values to find the most common range
    min_depth = np.min(valid_depths)
    max_depth = np.max(valid_depths)
    
    # Create bins
    num_bins = max(1, int((max_depth - min_depth) / bin_size))
    hist, bin_edges = np.histogram(valid_depths, bins=num_bins)
    
    # Find the bin with the most occurrences
    most_common_bin_idx = np.argmax(hist)
    
    # Get the center of the most common bin
    bin_center = (bin_edges[most_common_bin_idx] + bin_edges[most_common_bin_idx + 1]) / 2
    
    # Alternative approach: find values within the most common bin and return their median
    bin_start = bin_edges[most_common_bin_idx]
    bin_end = bin_edges[most_common_bin_idx + 1]
    values_in_bin = valid_depths[(valid_depths >= bin_start) & (valid_depths < bin_end)]
    
    if len(values_in_bin) > 0:
        return np.median(values_in_bin)  # Use median of the most common bin for better accuracy
    else:
        return bin_center

def preprocess_reid_crop(crop, input_shape=(224,224)):
    # Assumes crop is a BGR numpy array from OpenCV
    img = cv2.resize(crop, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img

def hailo_reid_embedder(crops):
    with hailo_activation_lock:
        with reid_network_group.activate(reid_network_group_params):
            with hpf.InferVStreams(reid_network_group, reid_inp_vstreams, reid_out_vstreams) as infer_pipeline:
                
                # Preprocess all crops and stack into batch
                batch_input = np.stack([preprocess_reid_crop(crop, (224, 224)) for crop in crops], axis=0)

                # Prepare input dictionary (note: input_data shape is [batch, ...])
                input_data = {reid_input_info.name: batch_input}

                # Perform a single batched inference
                outputs = infer_pipeline.infer(input_data)

                # Extract the output tensor
                output_data = outputs[reid_output_info.name] if reid_output_info.name in outputs else list(outputs.values())[0]

                # Reshape to (batch_size, embedding_dim)
                embeddings = output_data.reshape((len(crops), -1))

    return embeddings

camera_height_in_mm = (persons_height * 0.5) * 10
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
    nomalize_boundary = 0.8
    d_normalized = (((d - min_x) / (max_x - min_x)) * nomalize_boundary)

    if d_normalized < 0 or d_normalized > nomalize_boundary:
        return (0, 0)

    power = nomalize_boundary - d_normalized
    return (1, 40)

# def init_whisper_pipeline(vdevice=None):

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

vdevice = None

if vdevice is None:
    params = hpf.VDevice.create_params()
    #params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
    vdevice = hpf.VDevice(params)

# Load Hailo YOLO model once at startup
hef = hpf.HEF(HEF_PATH)
cfg = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
network_group = vdevice.configure(hef, cfg)[0]
network_group_params = network_group.create_params()
input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]

inp_vstreams = hpf.InputVStreamParams.make_from_network_group(
    network_group, quantized=True, format_type=hpf.FormatType.AUTO)
out_vstreams = hpf.OutputVStreamParams.make_from_network_group(
    network_group, quantized=True, format_type=hpf.FormatType.AUTO)

# Load Hailo ReID model once at startup
reid_hef = hpf.HEF(REID_HEF_PATH)
reid_cfg = hpf.ConfigureParams.create_from_hef(reid_hef, interface=hpf.HailoStreamInterface.PCIe)
reid_network_group = vdevice.configure(reid_hef, reid_cfg)[0]
reid_network_group_params = reid_network_group.create_params()
reid_input_info = reid_hef.get_input_vstream_infos()[0]
reid_output_info = reid_hef.get_output_vstream_infos()[0]

reid_inp_vstreams = hpf.InputVStreamParams.make_from_network_group(
    reid_network_group, quantized=True, format_type=hpf.FormatType.AUTO)
reid_out_vstreams = hpf.OutputVStreamParams.make_from_network_group(
    reid_network_group, quantized=True, format_type=hpf.FormatType.AUTO)


# Paths to your compiled Whisper models
# ENCODER_HEF = "resources/base-whisper-encoder-5s.hef"
# DECODER_HEF = "resources/base-whisper-decoder-fixed-sequence-matmul-split.hef"

# # Encoder Network Group
# encoder_hef = hpf.HEF(ENCODER_HEF)
# encoder_cfg = hpf.ConfigureParams.create_from_hef(encoder_hef, interface=hpf.HailoStreamInterface.PCIe)
# encoder_group = vdevice.configure(encoder_hef, encoder_cfg)[0]
# encoder_params = encoder_group.create_params()
# enc_input_info = encoder_hef.get_input_vstream_infos()[0]
# enc_output_info = encoder_hef.get_output_vstream_infos()[0]
# encoder_inputs = hpf.InputVStreamParams.make_from_network_group(encoder_group, quantized=True, format_type=hpf.FormatType.AUTO)
# encoder_outputs = hpf.OutputVStreamParams.make_from_network_group(encoder_group, quantized=True, format_type=hpf.FormatType.AUTO)

# # Decoder Network Group
# decoder_hef = hpf.HEF(DECODER_HEF)
# decoder_cfg = hpf.ConfigureParams.create_from_hef(decoder_hef, interface=hpf.HailoStreamInterface.PCIe)
# decoder_group = vdevice.configure(decoder_hef, decoder_cfg)[0]
# decoder_params = decoder_group.create_params()
# dec_input_info = decoder_hef.get_input_vstream_infos()[0]
# dec_output_info = decoder_hef.get_output_vstream_infos()[0]
# decoder_inputs = hpf.InputVStreamParams.make_from_network_group(decoder_group, quantized=True, format_type=hpf.FormatType.AUTO)
# decoder_outputs = hpf.OutputVStreamParams.make_from_network_group(decoder_group, quantized=True, format_type=hpf.FormatType.AUTO)


# === Initialize DeepSORT tracker ===
tracker = DeepSort(max_age=30, n_init=1, nms_max_overlap=0.5, max_cosine_distance=0.3)

processor = WhisperProcessor.from_pretrained("openai/whisper-base")

def decode_whisper_tokens(tokens):
    # filter to a Python list of ints
    token_ids = tokens.flatten().tolist()
    # skip special tokens and return plain text
    transcript = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    return transcript

init_speech_to_text(vdevice)

frame_count = 0
start_time = time.time()
tracking_frame_idx = 3

def identify_objects(rgbFrame):
    formatted_detections = []
    outputs = None
    # === Object Detection with Hailo-8 ===
    # Resize RGB frame to match YOLO input size (assuming 640x640)
    yolo_input = cv2.resize(rgbFrame, (640, 640))
    input_for_model = np.expand_dims(yolo_input, axis=0).astype(np.uint8)
    input_data = {input_info.name: input_for_model}
    with hailo_activation_lock:
        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, inp_vstreams, out_vstreams) as infer_pipeline:
                timer.log("Activate YOLO network")
                outputs = infer_pipeline.infer(input_data)
    output_data = outputs[output_info.name] if output_info.name in outputs else list(outputs.values())[0]
    timer.log("YOLO inference")

    for class_id, class_dets in enumerate(output_data[0]):
        if class_dets.shape[0] == 0:
            continue
        for det in class_dets:
            ymin, xmin, ymax, xmax, score = det
            if score < 0.5:
                continue
            # Scale coordinates back to original RGB frame size
            x1 = int(xmin * rgbFrame.shape[1])
            y1 = int(ymin * rgbFrame.shape[0])
            x2 = int(xmax * rgbFrame.shape[1])
            y2 = int(ymax * rgbFrame.shape[0])
            w, h = x2 - x1, y2 - y1
            class_name = COCO_CLASS_ID_MAP.get(class_id, None)
            if class_name is None:
                continue  # skip unknown class

            formatted_detections.append([[x1, y1, w, h], score, class_name])

    timer.log("YOLO postprocessing")
    return formatted_detections

def detect_objects(spatialData):   
    red_level_rois = []
    yellow_level_rois = []
    floor_level_rois = []
    for i, depthData in enumerate(spatialData):
        roi = depthData.config.roi
        roiRgb = roi.denormalize(width=rgbFrame.shape[1], height=rgbFrame.shape[0])
        # xmin, ymin = int(roiDepth.topLeft().x), int(roiDepth.topLeft().y)
        # xmax, ymax = int(roiDepth.bottomRight().x), int(roiDepth.bottomRight().y)
        xmin_rgb, ymin_rgb = int(roiRgb.topLeft().x), int(roiRgb.topLeft().y)
        xmax_rgb, ymax_rgb = int(roiRgb.bottomRight().x), int(roiRgb.bottomRight().y)

        coords = depthData.spatialCoordinates
        distance = math.sqrt(coords.x**2 + coords.y**2 + coords.z**2)
        height = coords.y
        color = (0, 255, 255)

        is_hazard = False
        head_level = False
        floor_level = False
        yellow_level = False

        if height < 0 and height <= (camera_height_in_mm * -1):
            color = (255, 0, 0)
            floor_level = True
        elif (camera_height_in_mm + height) <= (persons_height * 10) + 100:
            if (camera_height_in_mm + height) >= (persons_height * 10) - 100:
                head_level = True
                is_hazard = True
                # print("(camera_height_in_mm + height):", (camera_height_in_mm + height))
                # print("(persons_height * 10):", (persons_height * 10))
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
            else:
                yellow_level = True
        else:
            yellow_level = True


        # --- Sliding window logic ---
        hist = hazard_history.setdefault(i, [])
        hist.append(is_hazard)
        if len(hist) > WINDOW_LEN:
            hist.pop(0)
        # True hazard only if all last window frames are True
        is_hazard_windowed = (len(hist) == WINDOW_LEN and all(hist))

        roi = {
                'index': i,
                'roi_rect': (xmin_rgb, ymin_rgb, xmax_rgb, ymax_rgb),
                'distance': distance,
                'height': height,
                'roi': roi
            }
        if is_hazard_windowed:
            color = (0, 0, 255)
            red_level_rois.append(roi)
        elif floor_level:
            floor_level_rois.append(roi)
        else:
            yellow_level_rois.append(roi)
            

    return red_level_rois, yellow_level_rois, floor_level_rois

# Single Button Press
def rerun_red_level_identification():
    rgbFrame = None
    spatialData = None
    with global_data_lock:
        rgbFrame = global_rgb_frame
        spatialData = global_spatial_data

    formatted_detections = identify_objects(rgbFrame)
    red_level_rois, yellow_level_rois, floor_level_rois = detect_objects(spatialData)

    for det in formatted_detections:
        box, conf, class_name = det
        x1, y1, w, h = box
        bbox = (x1, y1, x1 + w, y1 + h)
        
        # Check if this detection intersects with any hazard ROI
        for hazard_roi in red_level_rois:
            if bbox_intersects_roi(bbox, hazard_roi['roi_rect']):
                # Get average distance in the bounding box
                avg_distance = get_distance_in_bbox(depthFrame, bbox)
                
                # Print warning every time (no tracking)
                detection_key = f"{class_name}_{x1}_{y1}_{w}_{h}"
                print(f"Hazard: object {class_name} at {avg_distance/1000:.1f} meters")
                speak_text(f"{class_name} {avg_distance/1000:.1f} meters")
                
                # Draw purple bounding box
                if SHOW_WINDOWS:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                    cv2.putText(rgbFrame, f"{class_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)
                break

def single_click():
    # stop_non_priority_speech()
    # stop_priority_speech()
    rerun_red_level_identification()

def double_click():
    # stop_non_priority_speech()
    # stop_priority_speech()
    speak_text("Please stand straight and point camera towards direction you want described")
    time.sleep(10)
    # stop_non_priority_speech()
    # stop_priority_speech()
    speak_text("Analyzing")
    describe_scene_in_detail()

# print("Expected input shape:", enc_input_info.shape)
# print("Expected dtype:", enc_input_info.format)  # Safer: prints enum like FormatType.UINT8s

# dec_input_infos = decoder_hef.get_input_vstream_infos()
# print("Decoder expects:", [inp.name for inp in dec_input_infos])


# audio_filename = "/tmp/recorded.wav"
# audio_fs = 16000  # Whisper expects 16kHz
session = None
def hold():
    global session
    # stop_priority_speech()
    # stop_non_priority_speech()
    speak_text("Listening")
    # hailo_activation_lock.acquire()
    session = start_listening(max_duration=60)
    # global audio_recording
    # speak_text("Listening")
    # hailo_activation_lock.acquire()

    # Start recording in background thread
    # audio_recording = sd.rec(int(60 * audio_fs), samplerate=audio_fs, channels=1, dtype='int16')

def triple_click():
    # stop_priority_speech()
    # stop_non_priority_speech()
    return

def hold_release():
    global session
    if session is None:
        return
    # stop_non_priority_speech()
    # stop_priority_speech()
    speak_text("Stop Listening")
    transcription = ""
    with hailo_activation_lock:
        transcription = session.finish()
        # speak_text(transcription)
        # hailo_activation_lock.release() 
    session = None

    transcription = transcription.lower()
    if "question" in transcription:          # check
        transcription = transcription.replace("question", "", 1)
        print("Questioning LLM:", transcription)
        question_llm(transcription)

    # # --- stop mic & create mel ---
    # sd.stop()
    # sf.write(audio_filename, audio_recording, audio_fs)
    # mel_input = preprocess_mel(audio_recording.flatten(), sr=audio_fs)

    # with hailo_activation_lock:
    #     # ---------- 1) ENCODER ----------
    #     with encoder_group.activate(encoder_params):
    #         with hpf.InferVStreams(encoder_group,
    #                                 encoder_inputs,
    #                                 encoder_outputs) as enc_pipe:
    #             latent = enc_pipe.infer(
    #                 {enc_input_info.name: np.ascontiguousarray(mel_input)}
    #             )[enc_output_info.name]   # shape ~ (1,1500,384)  dtype uint8

    #     # ---------- 2) PREP TOKENS ----------
    #     bos = processor.tokenizer.bos_token_id        # 50257
    #     token_buf = np.array([[bos]], dtype=np.int32) # (1,1) int32

    #     # ---------- 3) DECODER ----------
    #     # discover the exact input-vstream ordering *once*
    #     token_port, latent_port = None, None
    #     for info in decoder_hef.get_input_vstream_infos():
    #         # NEW
    #         print(f"[decoder] {info.name} expects {info.shape} {info.format}")
    #         if np.prod(info.shape) == token_buf.size:   # (1,1) -> 1 element
    #             token_port = info.name
    #         else:
    #             latent_port = info.name

    #     if token_port is None or latent_port is None:
    #         raise RuntimeError("Could not map decoder inputs automatically.")

    #     decoder_inputs_dict = {
    #         token_port:  np.ascontiguousarray(token_buf.astype(info.format.data_type.to_numpy_dtype())),
    #         latent_port: np.ascontiguousarray(latent.astype(np.uint8))
    #     }

    #     with decoder_group.activate(decoder_params):
    #         with hpf.InferVStreams(decoder_group,
    #                                 decoder_inputs,
    #                                 decoder_outputs) as dec_pipe:
    #             tokens = dec_pipe.infer(decoder_inputs_dict)[dec_output_info.name]

    # # ---------- 4) DECODE ----------
    # transcript = decode_whisper_tokens(tokens)
    # speak_text(transcript)



register_single_click(single_click)
register_double_click(double_click)
register_hold(hold)
register_hold_release(hold_release)
register_triple_click(triple_click)

# Vibrate motors for startup
vibrate_motors_by_region([(1, 2000), (1, 2000), (1, 2000)])
print("PROGRAM STARTED")

# Run pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    fontType = cv2.FONT_HERSHEY_TRIPLEX
    timer.log("Setup")

    while True:
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame()

        rgbFrame = rgbQueue.get().getCvFrame()
        spatialData = spatialCalcQueue.get().getSpatialLocations()
        with global_data_lock:
            global_rgb_frame = rgbFrame
            global_spatial_data = spatialData

        depth_downscaled = depthFrame[::4]
        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1) if np.any(depth_downscaled != 0) else 0
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        depthFrameColor = depthFrameColor[:, 10:-10]  # crop here

        timer.log("Pre-processing")

        formatted_detections = identify_objects(rgbFrame)

        # === Tracking ===
        tracks = []
        if enable_tracking and (frame_count % tracking_frame_idx == 0):
            if formatted_detections:
                object_chips = []
                for [[x1, y1, w, h], _, _] in formatted_detections:
                    x2 = x1 + w
                    y2 = y1 + h
                    if x1 < 0 or y1 < 0 or x2 > rgbFrame.shape[1] or y2 > rgbFrame.shape[0] or w <= 0 or h <= 0:
                        continue
                    chip = rgbFrame[y1:y2, x1:x2]
                    if chip.size == 0:
                        continue
                    object_chips.append(chip)

                if object_chips:
                    embeds = hailo_reid_embedder(object_chips)
                    tracks = tracker.update_tracks(formatted_detections, embeds=embeds)
                else:
                    tracks = tracker.update_tracks([], frame=rgbFrame)


        timer.log("Deepsort Tracking")

        # === Hazard Detection and Motor Control ===
        left_min = float('inf')
        middle_min = float('inf')
        right_min = float('inf')
        head_min = float('inf')
        hazard_rois = []

        region_config = [dist_to_power_and_time(0), dist_to_power_and_time(0), dist_to_power_and_time(0)]

        red_level_rois, yellow_level_rois, floor_level_rois = detect_objects(spatialData)
        # --- Haptic feedback logic based on depth zones ---
        for roi in red_level_rois:
            col = (roi['index'] % num_cols) + 1
            distance = roi['distance']
            height = roi['height']
            if DEBUG:
                print("1 <", (center_square - (center_rl_squares // 2)), "<", (center_square + (center_rl_squares // 2)), "<", num_cols)
                print("col:", col)
                print("Center square:", center_square)
            
            # if head_level:
            #     head_min = min(head_min, distance)
            if 1 <= col < (center_square - (center_rl_squares // 2)):
                left_min = min(left_min, distance)
            elif (center_square + (center_rl_squares // 2)) < col <= num_cols:
                right_min = min(right_min, distance)
            elif (center_square - (center_rl_squares // 2)) <= col <= (center_square + (center_rl_squares // 2)):
                middle_min = min(middle_min, distance)

            region_config = [
                dist_to_power_and_time(left_min),
                dist_to_power_and_time(middle_min),
                dist_to_power_and_time(right_min),
            ]
            
        # Draw rectangles on both frames
        if SHOW_WINDOWS:
            def display_squares(rois, color):
                for roi in rois:
                    _roi = roi['roi']
                    distance = roi['distance']
                    height = roi['height']
                    xmin_rgb = roi['roi_rect'][0]
                    ymin_rgb = roi['roi_rect'][1]
                    xmax_rgb = roi['roi_rect'][2]
                    ymax_rgb = roi['roi_rect'][3]
                    roiDepth = _roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

                    xmin, ymin = int(roiDepth.topLeft().x), int(roiDepth.topLeft().y)
                    xmax, ymax = int(roiDepth.bottomRight().x), int(roiDepth.bottomRight().y)
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.rectangle(rgbFrame, (xmin_rgb, ymin_rgb), (xmax_rgb, ymax_rgb), color, 2)
                    cv2.putText(depthFrameColor, "{:.1f}m".format(distance / 1000), (xmin + 5, ymin + 20), fontType, 0.5, color)
                    cv2.putText(rgbFrame, "{:.1f}m".format((camera_height_in_mm + height)/1000), (xmin_rgb + 5, ymin_rgb + 20), fontType, 0.5, color)
            display_squares(red_level_rois, color = (0, 0, 255))
            display_squares(yellow_level_rois, color = (0, 255, 255))
            display_squares(floor_level_rois, color = (255, 0, 0))

        # === Object Identification for Hazard ROIs ===
        if len(red_level_rois) > 0:
            if enable_tracking:
                # Use tracked objects
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue
                    
                    track_id = track.track_id
                    class_name = track.get_det_class()
                    ltrb = track.to_ltrb()
                    bbox = tuple(map(int, ltrb))
                    
                    # Check if this track intersects with any hazard ROI
                    for hazard_roi in red_level_rois:
                        if bbox_intersects_roi(bbox, hazard_roi['roi_rect']):
                            # Get average distance in the bounding box
                            avg_distance = get_distance_in_bbox(depthFrame, bbox)
                            
                            # Print warning once per track
                            if track_id not in warned_objects:# or abs(warned_objects[track_id] - avg_distance / 1000) >= 1.0:
                                print(f"Hazard: object {class_name} [ID {track_id}] at {avg_distance/1000:.1f} meters")
                                speak_text(f"{class_name} {avg_distance/1000:.1f} meters")
                                warned_objects[track_id] = avg_distance / 1000
                            
                            # Draw purple bounding box
                            if SHOW_WINDOWS:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                                cv2.putText(rgbFrame, f"ID {track_id}: {class_name}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)
                            break
            else:
                # Use raw detections
                for det in formatted_detections:
                    box, conf, class_name = det
                    x1, y1, w, h = box
                    bbox = (x1, y1, x1 + w, y1 + h)
                    
                    # Check if this detection intersects with any hazard ROI
                    for hazard_roi in hazard_rois:
                        if bbox_intersects_roi(bbox, hazard_roi['roi_rect']):
                            # Get average distance in the bounding box
                            avg_distance = get_distance_in_bbox(depthFrame, bbox)
                            
                            # Print warning every time (no tracking)
                            detection_key = f"{class_name}_{x1}_{y1}_{w}_{h}"
                            print(f"Hazard: object {class_name} at {avg_distance/1000:.1f} meters")
                            speak_text(f"{class_name} {avg_distance/1000:.1f} meters")
                            
                            # Draw purple bounding box
                            if SHOW_WINDOWS:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                                cv2.putText(rgbFrame, f"{class_name}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)
                            break

        timer.log("Object detection Postprocessing")
        # Clean up old warned objects (remove tracks that are no longer active)
        # if enable_tracking:
        #     active_track_ids = {track.track_id for track in tracks if track.is_confirmed() and track.time_since_update == 0}
        #     warned_objects = {tid: dist for tid, dist in warned_objects.items() if tid in active_track_ids}

        # Activate motors
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

        timer.print_log_line()  # Print this frame's timings
        timer.reset()

controller.close()
stop_speech_to_text()
