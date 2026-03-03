#!/usr/bin/env python3
import time
import cv2
import depthai as dai
import requests

SERVER_URL = "http://https://canepilotaivisionserver.aalwan.net/detect"  # <-- change
TIMEOUT_S = 10
SHOW = False

def is_valid_jpeg(jpeg_bytes: bytes) -> bool:
    # JPEG must start with FF D8 and end with FF D9
    return (
        isinstance(jpeg_bytes, (bytes, bytearray)) and
        len(jpeg_bytes) > 4 and
        jpeg_bytes[0:2] == b"\xff\xd8" and
        jpeg_bytes[-2:] == b"\xff\xd9"
    )

def build_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(30)

    # Use preview to reduce bandwidth/latency
    cam.preview.setSize(640, 360)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("frames")
    cam.preview.link(xout.input)

    return pipeline

def main():
    pipeline = build_pipeline()

    sess = requests.Session()

    with dai.Device(pipeline) as dev:
        q = dev.getOutputQueue("frames", maxSize=4, blocking=False)

        last_post = 0.0
        post_interval = 0.1  # seconds; adjust to control rate (0.1 = 10 FPS)

        while True:
            pkt = q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue

            frame = pkt.getCvFrame()  # BGR numpy image (safe, not JPEG bytes)

            # Encode to JPEG on host (guarantees a complete JPEG buffer)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                print("JPEG encode failed")
                continue
            jpg = buf.tobytes()

            # Validate before sending (prevents posting garbage)
            if not is_valid_jpeg(jpg):
                print(f"Invalid JPEG produced? len={len(jpg)} head={jpg[:8].hex()} tail={jpg[-8:].hex()}")
                continue

            # Throttle POST rate if needed
            now = time.time()
            if now - last_post < post_interval:
                if SHOW:
                    cv2.imshow("preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            last_post = now

            try:
                # Most common: multipart/form-data upload
                files = {"file": ("f.jpg", jpg, "image/jpeg")}
                data={"enable_tracking": "false"},
                r = sess.post(SERVER_URL, files=files, data=data, timeout=TIMEOUT_S)
                r.raise_for_status()

                # If server returns JSON:
                ctype = r.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    data = r.json()
                    print("Server JSON:", data)
                else:
                    # Or plain text
                    print("Server:", r.text[:300])

            except requests.RequestException as e:
                print("POST failed:", e)

            if SHOW:
                cv2.imshow("preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    main()
