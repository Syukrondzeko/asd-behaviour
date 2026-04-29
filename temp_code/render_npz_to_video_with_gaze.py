"""
Render FLAME coefficients (.npz) to video, with gaze injection.

If the .npz contains a 'gaze' key of shape (n_persons, n_frames, 3),
the gaze vectors are converted to eye rotation axis-angles and passed as
eye_pose_params directly to flame.forward() — the correct FLAME API.

In this project pose is 6-dim: [global_rot(3), jaw_rot(3)].
Eye rotation is a separate eye_pose_params: [left_eye_aa(3), right_eye_aa(3)]
where each 3-vec is an axis-angle (rotation around X = pitch).
"""
import argparse
import tempfile
from pathlib import Path
import sys
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _find_project_root() -> Path:
    """
    Resolve the fer-to-feg project root without hardcoding.

    Priority:
      1. FER_TO_FEG_ROOT environment variable
      2. Walk upward from __file__ looking for the 'models/flame.py' marker
    """
    env = os.environ.get('FER_TO_FEG_ROOT')
    if env:
        return Path(env).resolve()

    marker = Path('models') / 'flame.py'
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        candidate = start
        for _ in range(10):
            if (candidate / marker).exists():
                return candidate
            candidate = candidate.parent

    raise RuntimeError(
        "Could not locate fer-to-feg project root. "
        "Set the FER_TO_FEG_ROOT environment variable to its absolute path."
    )


PROJECT_ROOT = None


def _ensure_project_root(fer_root: str = None) -> Path:
    """Resolve and cache project root before importing fer-to-feg modules."""
    global PROJECT_ROOT
    if PROJECT_ROOT is not None:
        return PROJECT_ROOT

    if fer_root:
        PROJECT_ROOT = Path(fer_root).resolve()
        marker = PROJECT_ROOT / 'models' / 'flame.py'
        if not marker.exists():
            raise RuntimeError(
                f"Invalid --fer-root path: {PROJECT_ROOT} (missing {marker})"
            )
    else:
        PROJECT_ROOT = _find_project_root()

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return PROJECT_ROOT


# ---------------------------------------------------------------------------
# Gaze helpers
# ---------------------------------------------------------------------------

def gaze_to_eye_pose_params(gaze: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) unit gaze vectors to FLAME eye_pose_params (N, 6).

    gaze: [gaze_x, gaze_y, gaze_z] in camera space (+x right, +y up, -z forward)

    FLAME eye_pose_params layout: [left_eye_aa(3), right_eye_aa(3)]
    Each 3-vec is an axis-angle. Rotation around the X axis = vertical gaze:
      pitch = atan2(-gy, -gz)
      aa    = [pitch, 0, 0]
    Horizontal gaze (yaw) maps to rotation around Y:
      yaw   = atan2(-gx, -gz)
      aa    = [pitch, yaw, 0]
    """
    gx, gy, gz = gaze[:, 0], gaze[:, 1], gaze[:, 2]
    pitch = np.arctan2(-gy, -gz).astype(np.float32)   # vertical
    yaw   = np.arctan2(-gx, -gz).astype(np.float32)   # horizontal
    zeros = np.zeros_like(pitch)
    eye_aa = np.stack([pitch, yaw, zeros], axis=1)     # (N, 3)
    # Same rotation for both eyes
    eye_pose = np.concatenate([eye_aa, eye_aa], axis=1)  # (N, 6)
    return eye_pose


def build_eye_pose_params(raw_gaze, person_idx: int, is_2d: bool,
                          n_frames: int, device) -> torch.Tensor:
    """
    Returns eye_pose_params tensor (n_frames, 6) on device.
    If raw_gaze is None returns zeros (FLAME default = eyes forward).
    """
    if raw_gaze is None:
        return torch.zeros(n_frames, 6, device=device)
    g = raw_gaze[person_idx] if is_2d else raw_gaze[0]  # (n_frames, 3)
    n = min(n_frames, len(g))
    ep = np.zeros((n_frames, 6), dtype=np.float32)
    ep[:n] = gaze_to_eye_pose_params(g[:n])
    return torch.from_numpy(ep).to(device)


def vertices_with_eye_pose(flame, shape_coef, exp_coef, pose_coef,
                           eye_pose_params, ignore_global_rot=False,
                           batch_size=512):
    """
    Batch-call flame.forward() with explicit eye_pose_params.
    Returns (n_frames, V, 3) numpy array.
    """
    n = shape_coef.shape[0]
    verts_list = []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        v, _, _ = flame(
            shape_coef[sl], exp_coef[sl], pose_coef[sl],
            eye_pose_params=eye_pose_params[sl],
            pose2rot=True,
            ignore_global_rot=ignore_global_rot,
            return_lm2d=False, return_lm3d=False,
        )
        verts_list.append(v.detach().cpu())
    return torch.cat(verts_list, dim=0).numpy()


def draw_gaze_arrow(frame, gaze_vector, arrow_length=60, color=(0, 255, 0), thickness=2):
    """
    Draw gaze direction arrow overlay on rendered frame.

    Args:
        frame: (H, W, 3) RGB image
        gaze_vector: (3,) unit gaze direction in camera space [gx, gy, gz]
        arrow_length: arrow length in pixels
        color: BGR tuple (default: green)
        thickness: line thickness

    Returns:
        frame with arrow drawn
    """
    # Make a writable copy (renderer output is read-only)
    frame = frame.copy()
    h, w = frame.shape[:2]
    center = np.array([w // 2, h // 2])  # Head roughly centered in render

    # Normalize gaze
    gaze_n = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)

    # Project gaze to 2D screen space:
    # Camera space: +x right, +y up, -z forward
    # Image space: +x right, +y down
    arrow_offset = np.array([
        -gaze_n[0] * arrow_length,   # match demo.py: dx = -length * gaze_x
        -gaze_n[1] * arrow_length    # match demo.py: dy = -length * gaze_y
    ])
    arrow_end = center + arrow_offset.astype(int)

    # Draw arrow
    cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(arrow_end),
                    color, thickness, tipLength=0.3)
    return frame


# ---------------------------------------------------------------------------
# Original helpers (unchanged)
# ---------------------------------------------------------------------------

def find_audio_file(npz_path, person_type='speaker'):
    npz_path = Path(npz_path)
    person_dir = npz_path.parent.name
    filename_stem = npz_path.stem
    base_path = npz_path.parent.parent.parent
    audio_dir = base_path / 'audio' / person_dir
    for ext in ['.flac', '.wav', '.mp3', '.aac', '.m4a']:
        audio_file = audio_dir / f"{filename_stem}_{person_type}{ext}"
        if audio_file.exists():
            return audio_file
    return None


# ---------------------------------------------------------------------------
# Main render function — gaze-aware
# ---------------------------------------------------------------------------

def render_npz_to_video(npz_path, output_path, audio_path=None, texture_path=None,
                        fps=25, render_speaker=True, render_listener=True,
                        black_bg=False, size=(640, 640), auto_audio=True,
                        apply_gaze=True, gaze_arrow_length=80, gaze_arrow_color=(0, 255, 0),
                        draw_gaze_arrow_overlay=True, output_role_suffix=None):
    """
    Render FLAME coefficients from .npz file to video(s), optionally applying
    gaze stored in the 'gaze' key of the npz and drawing gaze arrows.

    Args:
        apply_gaze (bool): If True and 'gaze' key exists, apply eye_pose_params.
        gaze_arrow_length (int): Arrow length in pixels.
        gaze_arrow_color (tuple): BGR color tuple.
        draw_gaze_arrow_overlay (bool): If True, draw gaze arrow on frames.
    """
    project_root = _ensure_project_root()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading FLAME model...")
    from models.flame import FLAME, FLAMEConfig
    flame = FLAME(FLAMEConfig)
    flame.to(device)
    flame.eval()
    faces = flame.faces_tensor.detach().cpu().numpy()

    print("Loading renderer...")
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
    from psbody.mesh import Mesh
    from utils.renderer import MeshRenderer

    uv_coords = np.load(project_root / 'models/data/uv_coords.npz')
    renderer = MeshRenderer(size, black_bg=black_bg)

    print(f"Loading coefficients from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    print(f"\nCoefficient shapes:")
    for k in ['shape', 'exp', 'pose', 'gaze']:
        if k in data:
            print(f"  {k}: {data[k].shape}")

    is_2d_format = data['shape'].ndim == 3
    n_persons = data['shape'].shape[0] if is_2d_format else 1

    # Load gaze if available and requested
    raw_gaze = None
    if apply_gaze and 'gaze' in data:
        raw_gaze = data['gaze']
        print(f"\n[gaze] Found gaze key, shape={raw_gaze.shape} — will inject into eye pose")
    else:
        if apply_gaze:
            print("\n[gaze] No 'gaze' key in npz — rendering without gaze injection")
        else:
            print("\n[gaze] Gaze injection disabled via --no-gaze flag")

    if is_2d_format:
        if n_persons == 1:
            print("\nData format: 2D tensors but single person only (n_persons=1)")
        else:
            print(f"\nData format: 2D multi-person (n_persons={n_persons})")
    else:
        print("\nData format: 1D (single person)")

    to_render = []
    if is_2d_format:
        if n_persons == 1:
            # Many pipelines store single-person sequences as (1, N, D).
            # In that case both speaker/listener flags map to index 0.
            if render_speaker:
                to_render.append(('speaker', 0))
            if render_listener:
                to_render.append(('listener', 0))
        else:
            if render_speaker:
                to_render.append(('speaker', 0))
            if render_listener:
                to_render.append(('listener', 1))
    else:
        if render_listener:
            to_render.append(('listener', None))
        else:
            print("[WARNING] 1D format but render_listener=False. Nothing to render.")
            return

    output_path = Path(output_path)

    for person_type, person_idx in to_render:
        print(f"\n{'='*60}")
        print(f"Rendering {person_type}")
        print(f"{'='*60}")

        # Extract coefficients
        if is_2d_format:
            shape_coef = torch.from_numpy(data['shape'][person_idx]).float().to(device)
            exp_coef   = torch.from_numpy(data['exp'][person_idx]).float().to(device)
            pose_coef  = torch.from_numpy(data['pose'][person_idx]).float().to(device)
        else:
            shape_coef = torch.from_numpy(data['shape']).float().to(device)
            exp_coef   = torch.from_numpy(data['exp']).float().to(device)
            pose_coef  = torch.from_numpy(data['pose']).float().to(device)

        num_frames = shape_coef.shape[0]
        print(f"Number of frames: {num_frames}")
        print(f"Duration: {num_frames / fps:.2f} seconds")

        # Build eye_pose_params from gaze (zeros if no gaze available)
        eye_pose_params = build_eye_pose_params(
            raw_gaze, person_idx if is_2d_format else 0,
            is_2d=is_2d_format, n_frames=num_frames, device=device
        )
        if raw_gaze is not None:
            print(f"[gaze] eye_pose_params range: "
                  f"pitch [{eye_pose_params[:, 0].min():.3f}, {eye_pose_params[:, 0].max():.3f}] "
                  f"yaw [{eye_pose_params[:, 1].min():.3f}, {eye_pose_params[:, 1].max():.3f}] rad")

        # Convert to vertices calling FLAME directly with eye_pose_params
        print("Converting coefficients to vertices...")
        with torch.no_grad():
            verts_list = vertices_with_eye_pose(
                flame, shape_coef, exp_coef, pose_coef, eye_pose_params,
                ignore_global_rot=False
            )
        print(f"Vertices shape: {verts_list.shape}")

        # Texture
        texture = None
        if texture_path:
            texture = cv2.cvtColor(cv2.imread(str(texture_path)), cv2.COLOR_BGR2RGB)

        # Output path
        if len(to_render) > 1:
            out_file = output_path.parent / f"{output_path.stem}_{person_type}{output_path.suffix}"
        else:
            if output_role_suffix:
                out_file = output_path.parent / f"{output_path.stem}_{output_role_suffix}{output_path.suffix}"
            else:
                out_file = output_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Audio
        current_audio_path = None
        if audio_path == 'auto' or (auto_audio and audio_path is None):
            current_audio_path = find_audio_file(npz_path, person_type)
            if current_audio_path:
                print(f"Auto-detected audio: {current_audio_path}")
            else:
                print(f"No audio file found for {person_type}")
        elif audio_path is not None:
            current_audio_path = audio_path

        # Render frames
        print(f"Rendering to video: {out_file}")
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_file.parent, delete=False)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        center = np.mean(verts_list, axis=(0, 1))
        for frame_idx, verts in enumerate(tqdm(verts_list, desc=f'Rendering {person_type}')):
            mesh = Mesh(verts, faces)
            rendered, _ = renderer.render_mesh(mesh, center, tex_img=texture, tex_uv=uv_coords)

            # Overlay gaze arrow if enabled and gaze data is available
            if draw_gaze_arrow_overlay and raw_gaze is not None:
                g = raw_gaze[person_idx if is_2d_format else 0][frame_idx]  # (3,) gaze vector
                rendered = draw_gaze_arrow(rendered, g, arrow_length=gaze_arrow_length,
                                          color=gaze_arrow_color, thickness=3)

            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        writer.release()

        # Combine audio
        if current_audio_path is not None:
            print(f"Adding audio from {current_audio_path}")
            from utils.media import combine_video_and_audio, reencode_audio
            tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.aac', dir=out_file.parent, delete=False)
            reencode_audio(current_audio_path, tmp_audio_file.name)
            combine_video_and_audio(tmp_video_file.name, tmp_audio_file.name, out_file, copy_audio=False)
            os.unlink(tmp_audio_file.name)
        else:
            from utils.media import convert_video
            convert_video(tmp_video_file.name, out_file)
        os.unlink(tmp_video_file.name)

        print(f"✓ Saved: {out_file}")

    print(f"\n{'='*60}")
    print("Rendering complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Render FLAME .npz to video with gaze injection')
    parser.add_argument('--fer-root', type=str, default=None,
                        help='Path to fer-to-feg project root (overrides FER_TO_FEG_ROOT env var '
                             'and auto-detection)')
    parser.add_argument('--input',   type=str, required=True,  help='Input .npz file')
    parser.add_argument('--output',  type=str, required=True,  help='Output video path (.mp4)')
    parser.add_argument('--audio',   type=str, default=None,   help='Audio file or "auto"')
    parser.add_argument('--no-auto-audio', action='store_true')
    parser.add_argument('--texture', type=str, default=None)
    parser.add_argument('--fps',     type=int, default=25)
    parser.add_argument('--speaker', action='store_true', help='Render speaker only')
    parser.add_argument('--listener',action='store_true', help='Render listener only')
    parser.add_argument('--role', type=str, default=None, choices=['speaker', 'listener'],
                        help='Render exactly one role and append output filename suffix '
                             '(e.g., out.mp4 -> out_speaker.mp4)')
    parser.add_argument('--black-bg',action='store_true')
    parser.add_argument('--size',    type=int, nargs=2, default=[640, 640])
    parser.add_argument('--no-gaze', action='store_true',
                        help='Disable gaze injection even if gaze key exists in npz')
    parser.add_argument('--gaze-arrow-length', type=int, default=80,
                        help='Length of gaze arrow overlay in pixels (default: 80)')
    parser.add_argument('--gaze-arrow-color', type=str, default='green',
                        choices=['green', 'red', 'blue', 'yellow', 'cyan', 'magenta'],
                        help='Color of gaze arrow (default: green)')
    parser.add_argument('--no-gaze-arrow', action='store_true',
                        help='Disable gaze arrow overlay even if gaze data exists')
    args = parser.parse_args()

    _ensure_project_root(args.fer_root)

    # Color mapping
    color_map = {
        'green':   (0, 255, 0),
        'red':     (0, 0, 255),
        'blue':    (255, 0, 0),
        'yellow':  (0, 255, 255),
        'cyan':    (255, 255, 0),
        'magenta': (255, 0, 255),
    }
    arrow_color = color_map[args.gaze_arrow_color]

    output_role_suffix = None
    if args.role is not None:
        render_speaker = args.role == 'speaker'
        render_listener = args.role == 'listener'
        output_role_suffix = args.role
    else:
        render_speaker = not (args.listener and not args.speaker)
        render_listener = not (args.speaker and not args.listener)

    audio_path = args.audio
    if audio_path is None and not args.no_auto_audio:
        audio_path = 'auto'

    render_npz_to_video(
        args.input,
        args.output,
        audio_path=audio_path,
        texture_path=args.texture,
        fps=args.fps,
        render_speaker=render_speaker,
        render_listener=render_listener,
        black_bg=args.black_bg,
        size=tuple(args.size),
        auto_audio=not args.no_auto_audio,
        apply_gaze=not args.no_gaze,
        gaze_arrow_length=args.gaze_arrow_length,
        gaze_arrow_color=arrow_color,
        draw_gaze_arrow_overlay=not args.no_gaze_arrow,
        output_role_suffix=output_role_suffix,
    )


if __name__ == '__main__':
    main()
