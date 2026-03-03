#!/usr/bin/env python3
import argparse
import os
import time
import cv2
import depthai as dai
import requests

SERVER_URL_DEFAULT = "https://canepilotaivisionserver.aalwan.net/detect"
TIMEOUT_S_DEFAULT = 10

def is_valid_jpeg(jpeg_bytes: bytes) -> bool:
    return (
        isinstance(jpeg_bytes, (bytes, bytearray)) and
        len(jpeg_bytes) > 4 and
        jpeg_bytes[0:2] == b"\xff\xd8" and
        jpeg_bytes[-2:] == b"\xff\xd9"
    )

def build_rgb_only_pipeline():
    """
    DepthAI 3: Camera node (replaces deprecated ColorCamera), 640x400 BGR output.
    """
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    camera_output = camRgb.requestOutput(
        (640, 400),
        type=dai.ImgFrame.Type.BGR888p,
    )
    rgb_queue = camera_output.createOutputQueue(maxSize=4, blocking=True)
    return pipeline, rgb_queue


def capture_one_jpeg(rgb_queue, quality: int = 90) -> tuple[bytes, any]:
    """
    Returns (jpg_bytes, bgr_frame)
    """
    # Grab one frame (blocking)
    pkt = rgb_queue.get()
    frame = pkt.getCvFrame()  # BGR np.ndarray

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")

    jpg = buf.tobytes()
    if not is_valid_jpeg(jpg):
        raise RuntimeError(f"Invalid JPEG produced (len={len(jpg)})")

    return jpg, frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captured.jpg", help="Output JPEG path")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100")
    ap.add_argument("--show", action="store_true", help="Show captured frame in a window")
    ap.add_argument("--send", action="store_true", help="POST the captured JPEG to server")
    ap.add_argument("--server", default=SERVER_URL_DEFAULT, help="Server /detect URL")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S_DEFAULT, help="HTTP timeout seconds")
    args = ap.parse_args()

    pipeline, rgb_queue = build_rgb_only_pipeline()
    pipeline.start()
    try:
        # small warmup (optional): discard a couple frames so exposure settles a bit
        warmup_start = time.time()
        while time.time() - warmup_start < 0.2:
            if rgb_queue.has():
                _ = rgb_queue.get()

        jpg, frame = capture_one_jpeg(rgb_queue, quality=args.quality)
    finally:
        pipeline.stop()

    # Save JPEG
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(jpg)
    print(f"Saved: {os.path.abspath(args.out)} ({len(jpg)} bytes)")

    # Optional show
    if args.show:
        cv2.imshow("captured", frame)
        print("Press any key to exit preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optional send
    if args.send:
        sess = requests.Session()
        files = {"file": (os.path.basename(args.out), jpg, "image/jpeg")}
        try:
            r = sess.post(args.server, files=files, timeout=args.timeout)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "application/json" in ctype:
                print("Server JSON:", r.json())
            else:
                print("Server:", r.text[:1000])
        except requests.RequestException as e:
            print("POST failed:", e)

if __name__ == "__main__":
    main()
