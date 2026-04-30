#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    new_w = max(1, int(round(w * (target_h / float(h)))))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _black_frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def combine_side_by_side(first_video_path: str, second_video_path: str, output_path: str) -> None:
    cap1 = cv2.VideoCapture(first_video_path)
    cap2 = cv2.VideoCapture(second_video_path)

    if not cap1.isOpened():
        raise RuntimeError(f"Cannot open first video: {first_video_path}")
    if not cap2.isOpened():
        raise RuntimeError(f"Cannot open second video: {second_video_path}")

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = fps1 if fps1 and fps1 > 0 else (fps2 if fps2 and fps2 > 0 else 25.0)

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if min(w1, h1, w2, h2) <= 0:
        raise RuntimeError("Invalid input video dimensions.")

    target_h = max(h1, h2)
    out_w1 = int(round(w1 * (target_h / float(h1))))
    out_w2 = int(round(w2 * (target_h / float(h2))))
    out_size = (out_w1 + out_w2, target_h)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {output_path}")

    black_left = _black_frame(target_h, out_w1)
    black_right = _black_frame(target_h, out_w2)

    try:
        while True:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()

            if not ok1 and not ok2:
                break

            if ok1:
                left = _resize_to_height(frame1, target_h)
                if left.shape[1] != out_w1:
                    left = cv2.resize(left, (out_w1, target_h), interpolation=cv2.INTER_AREA)
            else:
                left = black_left

            if ok2:
                right = _resize_to_height(frame2, target_h)
                if right.shape[1] != out_w2:
                    right = cv2.resize(right, (out_w2, target_h), interpolation=cv2.INTER_AREA)
            else:
                right = black_right

            combined = np.concatenate([left, right], axis=1)
            writer.write(combined)
    finally:
        cap1.release()
        cap2.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine two videos side by side")
    parser.add_argument("--first_video_path", required=True, help="Path to first video")
    parser.add_argument(
        "--secon_video_path",
        "--second_video_path",
        dest="second_video_path",
        required=True,
        help="Path to second video",
    )
    parser.add_argument("--output_path", required=True, help="Path to output video")
    args = parser.parse_args()

    combine_side_by_side(args.first_video_path, args.second_video_path, args.output_path)
    print(f"Saved side-by-side video to: {args.output_path}")


if __name__ == "__main__":
    main()
