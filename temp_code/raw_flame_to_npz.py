"""
Convert EMOCA per-frame output directories into a single .npz file.

EMOCA output structure:
  <emoca_dir>/
    000001_000/
      shape.npy    (100,)
      exp.npy      (50,)
      pose.npy     (6,)
      eye_pose.npy (6,)
      tex.npy      (50,)
      detail.npy   (128,)
    000002_000/
      ...

Target .npz structure (single person / listener):
  shape:    (1, N, 100)  float32
  exp:      (1, N,  50)  float32
  pose:     (1, N,   6)  float32

Usage:
  docker exec muhamad_multipar conda run -n diffposetalkprod python \
    /home/muhamadsy/gaze3d/temp_code/raw_flame_to_npz.py \
    --input  /home/muhamadsy/gaze3d/sample_flame/train/sample_flame/Person_0/000/EMOCA_v2_lr_mse_20 \
    --output /home/muhamadsy/gaze3d/sample_input/000_listener.npz
"""

import argparse
import numpy as np
from pathlib import Path


def convert(input_dir: str, output_path: str):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect and sort frame directories — match NNNNNN_NNN pattern only
    import re
    frame_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and re.fullmatch(r'\d+_\d+', d.name)
    ])
    if not frame_dirs:
        raise RuntimeError(f"No frame subdirectories found in {input_dir}")
    print(f"Found {len(frame_dirs)} frames in {input_dir}")

    # Stack per-frame arrays
    shape_list = []
    exp_list   = []
    pose_list  = []

    for fd in frame_dirs:
      shape_list.append(np.load(fd / 'shape.npy'))
      exp_list.append(np.load(fd / 'exp.npy'))
      pose_list.append(np.load(fd / 'pose.npy'))

    # Stack to (N, dim) then add person dim → (1, N, dim)
    shape = np.stack(shape_list, axis=0)[np.newaxis]  # (1, N, 100)
    exp   = np.stack(exp_list,   axis=0)[np.newaxis]  # (1, N,  50)
    pose  = np.stack(pose_list,  axis=0)[np.newaxis]  # (1, N,   6)

    print(f"  shape: {shape.shape}  dtype={shape.dtype}")
    print(f"  exp:   {exp.shape}  dtype={exp.dtype}")
    print(f"  pose:  {pose.shape}  dtype={pose.dtype}")

    np.savez(output_path, shape=shape, exp=exp, pose=pose)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert EMOCA per-frame npy dirs to a single .npz')
    parser.add_argument('--input',  type=str, required=True,
                        help='EMOCA model output dir (contains 000001_000/, 000002_000/, ...)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npz file path')
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
