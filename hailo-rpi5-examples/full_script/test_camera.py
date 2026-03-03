#!/usr/bin/env python3
import time
import cv2
import depthai as dai
import os

SAVE_PATH = "captured.jpg"
SHOW = True  # show preview window

def is_valid_jpeg(jpeg_bytes: bytes) -> bool:
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

    cam.setPreviewSize(640, 360)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("frames")
    cam.preview.link(xout.input)

    return pipeline

def main():
    pipeline = build_pipeline()

    with dai.Device(pipeline) as dev:
        q = dev.getOutputQueue("frames", maxSize=4, blocking=True)

        print("Waiting for frame...")
        pkt = q.get()  # blocking until frame arrives
        frame = pkt.getCvFrame()

        # Encode to JPEG on host (same method as your server script)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            print("JPEG encode failed")
            return

        jpg = buf.tobytes()

        if not is_valid_jpeg(jpg):
            print("Invalid JPEG generated")
            return

        # Save to file
        with open(SAVE_PATH, "wb") as f:
            f.write(jpg)

        print(f"Saved image to {os.path.abspath(SAVE_PATH)} ({len(jpg)} bytes)")

        if SHOW:
            cv2.imshow("Captured Frame", frame)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
