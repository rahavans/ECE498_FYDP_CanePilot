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
    Matches your big script's RGB setup:
      camRgb.setPreviewSize(640, 400)
      camRgb.setInterleaved(False)
      camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
      camRgb.preview -> XLinkOut("rgb")
    """
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")

    camRgb.setPreviewSize(640, 400)
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    camRgb.preview.link(xoutRgb.input)
    return pipeline

def capture_one_jpeg(device: dai.Device, quality: int = 90) -> tuple[bytes, any]:
    """
    Returns (jpg_bytes, bgr_frame)
    """
    q = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)

    # Grab one frame (blocking)
    pkt = q.get()
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

    pipeline = build_rgb_only_pipeline()

    with dai.Device(pipeline) as device:
        # small warmup (optional): discard a couple frames so exposure settles a bit
        warmup_start = time.time()
        while time.time() - warmup_start < 0.2:
            _ = device.getOutputQueue(name="rgb", maxSize=4, blocking=False).tryGet()

        jpg, frame = capture_one_jpeg(device, quality=args.quality)

    # Save JPEG
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
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
    
