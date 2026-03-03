#!/usr/bin/env python3
import argparse
import sys
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

def is_valid_png(png_bytes: bytes) -> bool:
    return (
        isinstance(png_bytes, (bytes, bytearray)) and
        len(png_bytes) > 8 and
        png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    )

def download_image(url: str, timeout_s: int) -> tuple[bytes, str]:
    """
    Returns (content_bytes, content_type_guess)
    """
    r = requests.get(url, stream=True, timeout=timeout_s, headers={"User-Agent": "canepilot-tester/1.0"})
    r.raise_for_status()
    content = r.content

    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()

    # Some hosts lie about content-type; sniff magic bytes as backup.
    if is_valid_jpeg(content):
        return content, "image/jpeg"
    if is_valid_png(content):
        return content, "image/png"

    # Fall back to header if it looks like an image
    if ctype.startswith("image/"):
        return content, ctype

    raise ValueError(f"Downloaded content doesn't look like JPEG/PNG (Content-Type={ctype}, len={len(content)})")

def main():
    ap = argparse.ArgumentParser(description="Download a human image from the web and send it to /detect.")
    ap.add_argument("--server", default=SERVER_URL_DEFAULT, help="Detection endpoint URL")
    ap.add_argument(
        "--img-url",
        default="https://upload.wikimedia.org/wikipedia/commons/7/7e/Man_walking_in_Berlin.jpg",
        help="Direct URL to an image containing a person"
    )
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S_DEFAULT, help="HTTP timeout seconds")
    args = ap.parse_args()

    sess = requests.Session()

    try:
        img_bytes, mime = download_image(args.img_url, args.timeout)
    except Exception as e:
        print("Download failed:", e)
        sys.exit(2)

    # Pick a filename extension based on mime
    if mime == "image/jpeg":
        fname = "human.jpg"
    elif mime == "image/png":
        fname = "human.png"
    else:
        fname = "human.img"

    print(f"Downloaded {len(img_bytes)} bytes from {args.img_url}")
    print(f"Detected MIME: {mime}")

    try:
        files = {"file": (fname, img_bytes, mime)}
        r = sess.post(args.server, files=files, timeout=args.timeout)
        r.raise_for_status()

        ctype = r.headers.get("Content-Type", "")
        if "application/json" in ctype:
            print("Server JSON:", r.json())
        else:
            print("Server:", r.text[:1000])

    except requests.RequestException as e:
        print("POST failed:", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
