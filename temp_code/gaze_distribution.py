#!/usr/bin/env python3
"""
Compute gaze distribution and dominant viewing area.

Supports:
- NPZ input with a gaze key (default: gaze)
- CSV input with gaze_x, gaze_y, gaze_z columns

Output:
- Text summary in terminal
- Optional JSON summary file
- Optional 3x3 heatmap plot
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


H_LABELS = ["left", "center", "right"]
V_LABELS = ["up", "middle", "down"]


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return v / norm


def load_gaze_from_npz(npz_path: Path, gaze_key: str, person_index: int) -> np.ndarray:
    data = np.load(npz_path)
    if gaze_key not in data:
        raise KeyError(f"Key '{gaze_key}' not found in {npz_path}")

    g = data[gaze_key]
    # Expected common shapes:
    # (N, 3) or (P, N, 3)
    if g.ndim == 2:
        if g.shape[1] != 3:
            raise ValueError(f"Expected gaze shape (N,3), got {g.shape}")
        return g.astype(np.float32)

    if g.ndim == 3:
        n_persons = g.shape[0]
        idx = person_index
        if idx < 0 or idx >= n_persons:
            raise IndexError(f"person_index={idx} out of range for n_persons={n_persons}")
        out = g[idx]
        if out.shape[1] != 3:
            raise ValueError(f"Expected selected gaze shape (N,3), got {out.shape}")
        return out.astype(np.float32)

    raise ValueError(f"Unsupported gaze shape: {g.shape}")


def load_pose_from_npz(npz_path: Path, pose_key: str, person_index: int) -> np.ndarray:
    data = np.load(npz_path)
    if pose_key not in data:
        raise KeyError(f"Key '{pose_key}' not found in {npz_path}")

    p = data[pose_key]
    # Expected: (N,6) or (P,N,6). We use global rotation p[..., :3].
    if p.ndim == 2:
        if p.shape[1] < 3:
            raise ValueError(f"Expected pose shape (N,>=3), got {p.shape}")
        return p[:, :3].astype(np.float32)

    if p.ndim == 3:
        n_persons = p.shape[0]
        idx = person_index
        if idx < 0 or idx >= n_persons:
            raise IndexError(f"person_index={idx} out of range for n_persons={n_persons}")
        out = p[idx]
        if out.shape[1] < 3:
            raise ValueError(f"Expected selected pose shape (N,>=3), got {out.shape}")
        return out[:, :3].astype(np.float32)

    raise ValueError(f"Unsupported pose shape: {p.shape}")


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle array (N,3) to rotation matrices (N,3,3)."""
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    theta_safe = np.clip(theta, 1e-8, None)
    axis = aa / theta_safe

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    c = np.cos(theta[:, 0])
    s = np.sin(theta[:, 0])
    C = 1.0 - c

    R = np.zeros((aa.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = c + x * x * C
    R[:, 0, 1] = x * y * C - z * s
    R[:, 0, 2] = x * z * C + y * s
    R[:, 1, 0] = y * x * C + z * s
    R[:, 1, 1] = c + y * y * C
    R[:, 1, 2] = y * z * C - x * s
    R[:, 2, 0] = z * x * C - y * s
    R[:, 2, 1] = z * y * C + x * s
    R[:, 2, 2] = c + z * z * C

    # Near-zero angle -> identity
    small = (theta[:, 0] < 1e-8)
    if np.any(small):
        R[small] = np.eye(3, dtype=np.float32)
    return R


def compute_head_pose_distribution(global_rot_aa: np.ndarray, yaw_thresh_deg: float, pitch_thresh_deg: float) -> Dict:
    """
    Classify where head is facing from FLAME global rotation axis-angle.

    Practical mapping for FLAME global rot (in radians):
    - Y component controls left/right (negative=left, positive=right)
    - X component controls up/down (negative=up, positive=down)
    """
    rot_x_deg = np.degrees(global_rot_aa[:, 0])
    rot_y_deg = np.degrees(global_rot_aa[:, 1])

    # Left/right: negative -> left, positive -> right
    hx = classify_axis(rot_y_deg, yaw_thresh_deg)
    # Up/down: negative -> up, positive -> down
    vy = classify_axis(rot_x_deg, pitch_thresh_deg)

    row_map = {-1: 0, 0: 1, 1: 2}
    col_map = {-1: 0, 0: 1, 1: 2}
    counts = np.zeros((3, 3), dtype=np.int64)

    for i in range(global_rot_aa.shape[0]):
        r = row_map[int(vy[i])]
        c = col_map[int(hx[i])]
        counts[r, c] += 1

    total = int(global_rot_aa.shape[0])
    perc = (counts.astype(np.float64) / max(total, 1)) * 100.0
    flat_idx = int(np.argmax(counts))
    top_r, top_c = np.unravel_index(flat_idx, counts.shape)
    top_area = f"{V_LABELS[top_r]}-{H_LABELS[top_c]}"

    return {
        "total_frames": total,
        "thresholds": {
            "yaw_thresh_deg": float(yaw_thresh_deg),
            "pitch_thresh_deg": float(pitch_thresh_deg),
        },
        "counts_matrix": counts.tolist(),
        "percent_matrix": perc.round(3).tolist(),
        "top_area": top_area,
        "top_area_count": int(counts[top_r, top_c]),
        "top_area_percent": float(perc[top_r, top_c]),
        "stats": {
            "yaw_deg_min": float(rot_y_deg.min()),
            "yaw_deg_max": float(rot_y_deg.max()),
            "pitch_deg_min": float(rot_x_deg.min()),
            "pitch_deg_max": float(rot_x_deg.max()),
        },
    }


def load_gaze_from_csv(csv_path: Path) -> np.ndarray:
    rows: List[Tuple[float, float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required = {"gaze_x", "gaze_y", "gaze_z"}
        if not required.issubset(cols):
            raise ValueError(
                "CSV must contain columns: gaze_x, gaze_y, gaze_z. "
                f"Found: {sorted(cols)}"
            )

        for r in reader:
            try:
                gx = float(r["gaze_x"])
                gy = float(r["gaze_y"])
                gz = float(r["gaze_z"])
            except (TypeError, ValueError):
                continue
            rows.append((gx, gy, gz))

    if not rows:
        raise ValueError(f"No valid gaze rows found in {csv_path}")

    return np.asarray(rows, dtype=np.float32)


def classify_axis(values: np.ndarray, thresh: float) -> np.ndarray:
    # -1 = negative side, 0 = center band, +1 = positive side
    out = np.zeros(values.shape[0], dtype=np.int32)
    out[values < -thresh] = -1
    out[values > thresh] = 1
    return out


def compute_distribution(gaze: np.ndarray, x_thresh: float, y_thresh: float) -> Dict:
    gaze_n = _safe_normalize(gaze)

    # Match your arrow convention:
    # dx = -gx  (screen right is positive)
    # dy = -gy  (screen down is positive)
    sx = -gaze_n[:, 0]
    sy = -gaze_n[:, 1]

    hx = classify_axis(sx, x_thresh)  # -1 left, 0 center, +1 right
    vy = classify_axis(sy, y_thresh)  # -1 up, 0 middle, +1 down

    # Map to 3x3 matrix rows=[up,middle,down], cols=[left,center,right]
    row_map = {-1: 0, 0: 1, 1: 2}
    col_map = {-1: 0, 0: 1, 1: 2}

    counts = np.zeros((3, 3), dtype=np.int64)
    indices_by_area: Dict[str, List[int]] = {}

    for i in range(gaze_n.shape[0]):
        r = row_map[int(vy[i])]
        c = col_map[int(hx[i])]
        counts[r, c] += 1

        area = f"{V_LABELS[r]}-{H_LABELS[c]}"
        indices_by_area.setdefault(area, []).append(i)

    total = int(gaze_n.shape[0])
    perc = (counts.astype(np.float64) / max(total, 1)) * 100.0

    flat_idx = int(np.argmax(counts))
    top_r, top_c = np.unravel_index(flat_idx, counts.shape)
    top_area = f"{V_LABELS[top_r]}-{H_LABELS[top_c]}"

    yaw = np.degrees(np.arctan2(-gaze_n[:, 0], -gaze_n[:, 2]))
    pitch = np.degrees(np.arctan2(-gaze_n[:, 1], -gaze_n[:, 2]))

    return {
        "total_frames": total,
        "thresholds": {"x_thresh": float(x_thresh), "y_thresh": float(y_thresh)},
        "counts_matrix": counts.tolist(),
        "percent_matrix": perc.round(3).tolist(),
        "top_area": top_area,
        "top_area_count": int(counts[top_r, top_c]),
        "top_area_percent": float(perc[top_r, top_c]),
        "areas": indices_by_area,
        "stats": {
            "gx_min": float(gaze_n[:, 0].min()),
            "gx_max": float(gaze_n[:, 0].max()),
            "gy_min": float(gaze_n[:, 1].min()),
            "gy_max": float(gaze_n[:, 1].max()),
            "gz_min": float(gaze_n[:, 2].min()),
            "gz_max": float(gaze_n[:, 2].max()),
            "yaw_deg_min": float(yaw.min()),
            "yaw_deg_max": float(yaw.max()),
            "pitch_deg_min": float(pitch.min()),
            "pitch_deg_max": float(pitch.max()),
        },
    }


def format_summary_text(summary: Dict) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("GAZE DISTRIBUTION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total frames: {summary['total_frames']}")
    lines.append(
        f"Thresholds: x={summary['thresholds']['x_thresh']:.3f}, "
        f"y={summary['thresholds']['y_thresh']:.3f}"
    )
    lines.append("")

    lines.append("3x3 counts (rows: up/middle/down, cols: left/center/right):")
    counts = np.asarray(summary["counts_matrix"])
    lines.append(np.array2string(counts))
    lines.append("")

    lines.append("3x3 percent (%):")
    perc = np.asarray(summary["percent_matrix"])
    lines.append(np.array2string(perc, precision=2, suppress_small=False))
    lines.append("")

    lines.append(
        f"Most looked area: {summary['top_area']} "
        f"({summary['top_area_count']} frames, {summary['top_area_percent']:.2f}%)"
    )
    lines.append("")

    lines.append("Direction stats:")
    s = summary["stats"]
    lines.append(
        f"gx [{s['gx_min']:.4f}, {s['gx_max']:.4f}]  "
        f"gy [{s['gy_min']:.4f}, {s['gy_max']:.4f}]  "
        f"gz [{s['gz_min']:.4f}, {s['gz_max']:.4f}]"
    )
    lines.append(
        f"yaw_deg [{s['yaw_deg_min']:.2f}, {s['yaw_deg_max']:.2f}]  "
        f"pitch_deg [{s['pitch_deg_min']:.2f}, {s['pitch_deg_max']:.2f}]"
    )

    head = summary.get("head_pose")
    lines.append("")
    lines.append("-" * 60)
    lines.append("HEAD POSE (FACING DIRECTION)")
    lines.append("-" * 60)
    if head is None:
        lines.append("Head pose: unavailable (no NPZ pose key loaded)")
    else:
        lines.append(
            f"Thresholds: yaw={head['thresholds']['yaw_thresh_deg']:.1f} deg, "
            f"pitch={head['thresholds']['pitch_thresh_deg']:.1f} deg"
        )
        lines.append("")
        lines.append("3x3 counts (rows: up/middle/down, cols: left/center/right):")
        h_counts = np.asarray(head["counts_matrix"])
        lines.append(np.array2string(h_counts))
        lines.append("")
        lines.append("3x3 percent (%):")
        h_perc = np.asarray(head["percent_matrix"])
        lines.append(np.array2string(h_perc, precision=2, suppress_small=False))
        lines.append("")
        lines.append(
            f"Most facing area: {head['top_area']} "
            f"({head['top_area_count']} frames, {head['top_area_percent']:.2f}%)"
        )
        hs = head["stats"]
        lines.append(
            f"head_yaw_deg [{hs['yaw_deg_min']:.2f}, {hs['yaw_deg_max']:.2f}]  "
            f"head_pitch_deg [{hs['pitch_deg_min']:.2f}, {hs['pitch_deg_max']:.2f}]"
        )
    return "\n".join(lines)


def print_summary(summary: Dict) -> None:
    print(format_summary_text(summary))


def save_plot(summary: Dict, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for --save-plot. Install it first."
        ) from e

    counts = np.asarray(summary["counts_matrix"], dtype=np.float32)
    perc = np.asarray(summary["percent_matrix"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(perc, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percent")

    ax.set_xticks([0, 1, 2], labels=H_LABELS)
    ax.set_yticks([0, 1, 2], labels=V_LABELS)
    ax.set_xlabel("Horizontal")
    ax.set_ylabel("Vertical")
    ax.set_title("Gaze Distribution (3x3)")

    for r in range(3):
        for c in range(3):
            ax.text(
                c,
                r,
                f"{int(counts[r, c])}\n{perc[r, c]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute gaze distribution and dominant look area")
    p.add_argument("--input", required=True, help="Input gaze file (.npz or .csv)")
    p.add_argument("--format", choices=["auto", "npz", "csv"], default="auto")
    p.add_argument("--gaze-key", default="gaze", help="NPZ key for gaze array")
    p.add_argument("--pose-key", default="pose", help="NPZ key for pose array")
    p.add_argument("--person-index", type=int, default=0, help="Person index when gaze shape is (P,N,3)")
    p.add_argument("--x-thresh", type=float, default=0.08, help="Horizontal center-band threshold")
    p.add_argument("--y-thresh", type=float, default=0.08, help="Vertical center-band threshold")
    p.add_argument("--head-yaw-thresh-deg", type=float, default=10.0, help="Head yaw center-band threshold in degrees")
    p.add_argument("--head-pitch-thresh-deg", type=float, default=10.0, help="Head pitch center-band threshold in degrees")
    p.add_argument("--save-json", default=None, help="Optional output json path")
    p.add_argument("--save-text", default=None, help="Optional output text summary path")
    p.add_argument("--save-plot", default=None, help="Optional output plot path (png)")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for auto-saved reports (default: <project>/output_analytics)",
    )
    p.add_argument("--no-save", action="store_true", help="Disable auto-saving summary files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    fmt = args.format
    if fmt == "auto":
        ext = in_path.suffix.lower()
        if ext == ".npz":
            fmt = "npz"
        elif ext == ".csv":
            fmt = "csv"
        else:
            raise ValueError("Could not detect format. Use --format npz or --format csv")

    head_pose_summary = None
    if fmt == "npz":
        gaze = load_gaze_from_npz(in_path, args.gaze_key, args.person_index)
        try:
            head_global_rot = load_pose_from_npz(in_path, args.pose_key, args.person_index)
            head_pose_summary = compute_head_pose_distribution(
                head_global_rot,
                yaw_thresh_deg=args.head_yaw_thresh_deg,
                pitch_thresh_deg=args.head_pitch_thresh_deg,
            )
        except Exception as e:
            print(f"[head-pose] Skipping head pose analysis: {e}")
    else:
        gaze = load_gaze_from_csv(in_path)

    summary = compute_distribution(gaze, args.x_thresh, args.y_thresh)
    summary["head_pose"] = head_pose_summary
    text_summary = format_summary_text(summary)
    print(text_summary)

    # Auto-save reports unless disabled.
    if not args.no_save:
        if args.output_dir is not None:
            out_dir = Path(args.output_dir)
        else:
            out_dir = Path(__file__).resolve().parents[1] / "output_analytics"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = in_path.stem
        out_text = Path(args.save_text) if args.save_text else out_dir / f"{stem}_gaze_distribution.txt"
        out_json = Path(args.save_json) if args.save_json else out_dir / f"{stem}_gaze_distribution.json"

        out_text.parent.mkdir(parents=True, exist_ok=True)
        out_text.write_text(text_summary + "\n", encoding="utf-8")
        print(f"\nSaved text: {out_text}")

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON: {out_json}")
    else:
        if args.save_text:
            out_text = Path(args.save_text)
            out_text.parent.mkdir(parents=True, exist_ok=True)
            out_text.write_text(text_summary + "\n", encoding="utf-8")
            print(f"\nSaved text: {out_text}")
        if args.save_json:
            out_json = Path(args.save_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved JSON: {out_json}")

    if args.save_plot:
        out_plot = Path(args.save_plot)
        save_plot(summary, out_plot)
        print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()
