#!/usr/bin/env python3
import time
import cv2
import depthai as dai
import requests

SERVER_URL = "https://canepilotaivisionserver.aalwan.net/detect"
TIMEOUT_S = 10
SHOW_WINDOWS = False

# POST rate control
POST_FPS = 10.0                      # 10 uploads/sec
POST_INTERVAL = 1.0 / POST_FPS

JPEG_QUALITY = 85

def is_valid_jpeg(jpeg_bytes: bytes) -> bool:
    return (
        isinstance(jpeg_bytes, (bytes, bytearray)) and
        len(jpeg_bytes) > 4 and
        jpeg_bytes[0:2] == b"\xff\xd8" and
        jpeg_bytes[-2:] == b"\xff\xd9"
    )

def build_pipeline():
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(640, 400)  # same as your big script
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)  # same as your big script
    # Optional: camRgb.setFps(30)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    return pipeline

def main():
    pipeline = build_pipeline()
    sess = requests.Session()

    last_post = 0.0
    frame_idx = 0

    with dai.Device(pipeline) as device:
        rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            inRgb = rgbQueue.tryGet()
            if inRgb is None:
                time.sleep(0.001)
                continue

            rgbFrame = inRgb.getCvFrame()
            frame_idx += 1

            # Show preview (optional)
            if SHOW_WINDOWS:
                cv2.imshow("rgb", rgbFrame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Throttle upload rate
            now = time.time()
            if now - last_post < POST_INTERVAL:
                continue
            last_post = now

            # Encode to JPEG on host
            ok, buf = cv2.imencode(".jpg", rgbFrame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                print("JPEG encode failed")
                continue

            jpg = buf.tobytes()
            if not is_valid_jpeg(jpg):
                print(f"Invalid JPEG? len={len(jpg)} head={jpg[:8].hex()} tail={jpg[-8:].hex()}")
                continue

            try:
                files = {"file": ("frame.jpg", jpg, "image/jpeg")}
                r = sess.post(SERVER_URL, files=files, timeout=TIMEOUT_S)
                r.raise_for_status()

                ctype = r.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    print("Server JSON:", r.json())
                else:
                    print("Server:", r.text[:300])

            except requests.RequestException as e:
                print("POST failed:", e)

    if SHOW_WINDOWS:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
