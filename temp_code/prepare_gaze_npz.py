"""
Inject gaze vectors from gaze3d output CSV into a FLAME .npz file.

Reads gaze_x, gaze_y, gaze_z from the CSV (already unit vectors),
stores them as a 'gaze' key in the npz for use by the render script.

Usage:
  docker exec muhamad_multipar conda run -n diffposetalkprod python \\
    /home/muhamadsy/gaze3d/temp_code/prepare_gaze_npz.py \\
    --input  /home/muhamadsy/gaze3d/sample_input/000_speaker.npz \\
    --csv    /home/muhamadsy/gaze3d/output/000_speaker_image_predicted_gaze.csv \\
    --output /home/muhamadsy/gaze3d/sample_input/000_speaker_with_gaze.npz
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def inject_gaze_from_csv(input_npz: str, csv_path: str, output_npz: str):
    input_npz  = Path(input_npz)
    csv_path   = Path(csv_path)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(input_npz, allow_pickle=True))
    n_persons = data['shape'].shape[0]
    n_frames  = data['shape'].shape[1]
    print(f"Loaded npz: {input_npz}")
    print(f"  persons={n_persons}, frames={n_frames}")

    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {csv_path}  ({len(df)} rows)")

    # Sort by frame_id and extract gaze vectors
    df = df.sort_values('frame_id').reset_index(drop=True)
    gaze_np = df[['gaze_x', 'gaze_y', 'gaze_z']].values.astype(np.float32)  # (N_csv, 3)

    # Align to npz frame count
    n_align = min(n_frames, len(gaze_np))
    if len(gaze_np) != n_frames:
        print(f"  [WARNING] CSV has {len(gaze_np)} rows but npz has {n_frames} frames. "
              f"Using first {n_align} frames.")

    # Build gaze array (n_persons, n_frames, 3) — same gaze for all persons
    # since CSV is for one person (the speaker/listener tracked)
    gaze_all = np.zeros((n_persons, n_frames, 3), dtype=np.float32)
    gaze_all[:, :n_align, :] = gaze_np[:n_align]  # broadcast to all persons

    data['gaze'] = gaze_all
    print(f"  gaze key added: {gaze_all.shape}")
    print(f"  gaze_x range: [{gaze_np[:n_align, 0].min():.4f}, {gaze_np[:n_align, 0].max():.4f}]")
    print(f"  gaze_y range: [{gaze_np[:n_align, 1].min():.4f}, {gaze_np[:n_align, 1].max():.4f}]")
    print(f"  gaze_z range: [{gaze_np[:n_align, 2].min():.4f}, {gaze_np[:n_align, 2].max():.4f}]")

    np.savez(output_npz, **data)
    print(f"\nSaved to: {output_npz}")


def main():
    parser = argparse.ArgumentParser(
        description='Inject gaze CSV into FLAME .npz file')
    parser.add_argument('--input',  type=str, required=True,
                        help='Input .npz file (from raw_flame_to_npz.py)')
    parser.add_argument('--csv',    type=str, required=True,
                        help='Gaze CSV from gaze3d demo.py (*_image_predicted_gaze.csv)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npz file path')
    args = parser.parse_args()
    inject_gaze_from_csv(args.input, args.csv, args.output)


if __name__ == '__main__':
    main()
