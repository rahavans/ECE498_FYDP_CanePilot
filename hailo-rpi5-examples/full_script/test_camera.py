#!/usr/bin/env python3
import time
import cv2
import depthai as dai

SHOW_WINDOWS = False

GIF_PATH = "capture.gif"
GIF_FPS = 10.0                 # output GIF FPS
GIF_INTERVAL = 1.0 / GIF_FPS
MAX_GIF_SECONDS = 15           # cap length to avoid huge gifs (adjust as needed)

PREVIEW_W, PREVIEW_H = 640, 400

def build_pipeline():
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(PREVIEW_W, PREVIEW_H)
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    return pipeline

def write_gif_opencv(frames_bgr, path, fps):
    """
    Try writing GIF using OpenCV VideoWriter. Returns True if success.
    """
    if not frames_bgr:
        return False

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"GIF ")  # may or may not exist depending on build
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    if not writer.isOpened():
        return False

    for f in frames_bgr:
        writer.write(f)
    writer.release()
    return True

def write_gif_imageio(frames_bgr, path, fps):
    """
    Fallback GIF writer using imageio (pip install imageio).
    Converts BGR->RGB.
    """
    import imageio.v2 as imageio
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    duration = 1.0 / fps
    imageio.mimsave(path, frames_rgb, format="GIF", duration=duration)

def main():
    pipeline = build_pipeline()

    frames = []
    last_gif_add = 0.0
    start = time.time()

    print("Streaming... press Ctrl+C to stop and save GIF.")

    try:
        with dai.Device(pipeline) as device:
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            while True:
                inRgb = rgbQueue.tryGet()
                if inRgb is None:
                    time.sleep(0.001)
                    continue

                frame = inRgb.getCvFrame()

                if SHOW_WINDOWS:
                    cv2.imshow("rgb", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

                # Add frame to GIF buffer at GIF_FPS rate
                now = time.time()
                if now - last_gif_add >= GIF_INTERVAL:
                    last_gif_add = now
                    frames.append(frame.copy())

                # Optional cap so you don't eat RAM forever
                if now - start > MAX_GIF_SECONDS:
                    print(f"Reached MAX_GIF_SECONDS={MAX_GIF_SECONDS}, stopping...")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving GIF...")

    finally:
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()

    if not frames:
        print("No frames captured; not writing GIF.")
        return

    # Try OpenCV first, then imageio fallback
    ok = write_gif_opencv(frames, GIF_PATH, GIF_FPS)
    if ok:
        print(f"Saved GIF with OpenCV: {GIF_PATH} (frames={len(frames)}, fps={GIF_FPS})")
        return

    try:
        write_gif_imageio(frames, GIF_PATH, GIF_FPS)
        print(f"Saved GIF with imageio: {GIF_PATH} (frames={len(frames)}, fps={GIF_FPS})")
    except Exception as e:
        print("Failed to write GIF (OpenCV and imageio fallback both failed).")
        print("Error:", e)
        print("Fix: pip install imageio  (or install ffmpeg/imagemagick depending on your environment)")

if __name__ == "__main__":
    main()
