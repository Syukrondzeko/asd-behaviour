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
                           batch_size=512, return_lm2d=False):
    """
    Batch-call flame.forward() with explicit eye_pose_params.
    Returns (n_frames, V, 3) numpy array.
    """
    n = shape_coef.shape[0]
    verts_list = []
    lm2d_list = []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        v, lm2d, _ = flame(
            shape_coef[sl], exp_coef[sl], pose_coef[sl],
            eye_pose_params=eye_pose_params[sl],
            pose2rot=True,
            ignore_global_rot=ignore_global_rot,
            return_lm2d=return_lm2d, return_lm3d=False,
        )
        verts_list.append(v.detach().cpu())
        if return_lm2d:
            lm2d_list.append(lm2d.detach().cpu())

    verts_np = torch.cat(verts_list, dim=0).numpy()
    if return_lm2d:
        lm2d_np = torch.cat(lm2d_list, dim=0).numpy()
        return verts_np, lm2d_np
    return verts_np


def _project_points_to_image(points_3d, frame_shape, camera_pose, yfov, aspect_ratio=1.0):
    """Project world 3D points with the same perspective model used by renderer."""
    h, w = frame_shape[:2]

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]

    # World -> camera (OpenGL camera looks toward -Z)
    p_cam = (points_3d - t[None, :]) @ R
    z = -p_cam[:, 2]
    z = np.clip(z, 1e-4, None)

    tan_half = np.tan(float(yfov) * 0.5)
    x_ndc = p_cam[:, 0] / (z * tan_half * float(aspect_ratio))
    y_ndc = p_cam[:, 1] / (z * tan_half)

    x_px = (x_ndc + 1.0) * 0.5 * w
    y_px = (1.0 - (y_ndc + 1.0) * 0.5) * h
    return np.stack([x_px, y_px], axis=1)


def _draw_single_embedded_eye(frame, eye_pts_px, pupil_offset_px, max_radius_px):
    if eye_pts_px.shape[0] < 5:
        return frame

    eye_pts_i = np.round(eye_pts_px).astype(np.int32)
    eye_center = np.mean(eye_pts_px, axis=0)

    horizontal_span = np.linalg.norm(eye_pts_px[0] - eye_pts_px[3])
    radius = int(max(2, min(max_radius_px, round(horizontal_span * 0.22))))

    p = eye_center + pupil_offset_px
    p = np.round(p).astype(np.int32)

    eye_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(eye_mask, eye_pts_i, 255)

    overlay = frame.copy()
    iris_color = (55, 70, 95)
    pupil_color = (10, 10, 10)
    cv2.circle(overlay, (int(p[0]), int(p[1])), radius, iris_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (int(p[0]), int(p[1])), max(1, int(round(radius * 0.48))), pupil_color, -1,
               lineType=cv2.LINE_AA)

    alpha = 0.82
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    frame[eye_mask > 0] = blended[eye_mask > 0]
    return frame


def draw_gaze_eyeballs(frame, gaze_vector, lm2d_frame, eyeball_radius=26,
                       camera_pose=None, yfov=None, aspect_ratio=1.0,
                       pupil_gain=1.8):
    """Draw gaze-driven irises inside eye regions using FLAME 2D landmarks."""
    # Expected 68-point layout: left eye [36:42], right eye [42:48]
    if lm2d_frame is None or lm2d_frame.shape[0] < 48:
        return frame

    frame = frame.copy()
    if camera_pose is None or yfov is None:
        return frame
    lm_px = _project_points_to_image(
        lm2d_frame[:, :3],
        frame.shape,
        camera_pose=camera_pose,
        yfov=yfov,
        aspect_ratio=aspect_ratio,
    )

    left_eye = lm_px[36:42]
    right_eye = lm_px[42:48]

    gaze_n = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)

    def eye_local_offset(eye_pts):
        # Eye-local basis from landmarks so motion follows eye orientation on screen.
        eye_center = np.mean(eye_pts, axis=0)
        left_corner = eye_pts[0]
        right_corner = eye_pts[3]
        x_axis = right_corner - left_corner
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        x_axis = x_axis / x_norm
        y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

        # Project eye polygon extents into local axes.
        rel = eye_pts - eye_center[None, :]
        u = rel @ x_axis
        v = rel @ y_axis
        u_half = max(float(np.max(np.abs(u))), 1.0)
        v_half = max(float(np.max(np.abs(v))), 1.0)

        # Map gaze to local motion; keep arrow sign convention on x/y.
        du = (-gaze_n[0]) * u_half * 0.78 * float(pupil_gain)
        dv = (-gaze_n[1]) * v_half * 0.78 * float(pupil_gain)

        # Clamp to remain inside eye region visually.
        du = float(np.clip(du, -u_half * 0.9, u_half * 0.9))
        dv = float(np.clip(dv, -v_half * 0.9, v_half * 0.9))
        return (x_axis * du + y_axis * dv).astype(np.float32)

    left_offset = eye_local_offset(left_eye)
    right_offset = eye_local_offset(right_eye)

    frame = _draw_single_embedded_eye(frame, left_eye, left_offset, max_radius_px=eyeball_radius)
    frame = _draw_single_embedded_eye(frame, right_eye, right_offset, max_radius_px=eyeball_radius)
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
                        apply_gaze=True, gaze_eyeball_radius=26,
                        draw_gaze_eyeball_overlay=True, output_role_suffix=None,
                        pupil_gain=1.8):
    """
    Render FLAME coefficients from .npz file to video(s), optionally applying
    gaze stored in the 'gaze' key of the npz and drawing gaze eyeballs.

    Args:
        apply_gaze (bool): If True and 'gaze' key exists, apply eye_pose_params.
        gaze_eyeball_radius (int): Eyeball radius in pixels.
        draw_gaze_eyeball_overlay (bool): If True, draw gaze eyeballs on frames.
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
        # Avoid double-applying gaze when using eyeball overlay.
        gaze_for_eye_pose = None if draw_gaze_eyeball_overlay else raw_gaze
        eye_pose_params = build_eye_pose_params(
            gaze_for_eye_pose, person_idx if is_2d_format else 0,
            is_2d=is_2d_format, n_frames=num_frames, device=device
        )
        if raw_gaze is not None:
            print(f"[gaze] eye_pose_params range: "
                  f"pitch [{eye_pose_params[:, 0].min():.3f}, {eye_pose_params[:, 0].max():.3f}] "
                  f"yaw [{eye_pose_params[:, 1].min():.3f}, {eye_pose_params[:, 1].max():.3f}] rad")

        # Convert to vertices calling FLAME directly with eye_pose_params
        print("Converting coefficients to vertices...")
        with torch.no_grad():
            verts_list, lm2d_list = vertices_with_eye_pose(
                flame, shape_coef, exp_coef, pose_coef, eye_pose_params,
                ignore_global_rot=False,
                return_lm2d=True,
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

            # Overlay gaze eyeballs if enabled and gaze data is available
            if draw_gaze_eyeball_overlay and raw_gaze is not None:
                g = raw_gaze[person_idx if is_2d_format else 0][frame_idx]  # (3,) gaze vector
                rendered = draw_gaze_eyeballs(
                    rendered,
                    g,
                    lm2d_frame=lm2d_list[frame_idx],
                    eyeball_radius=gaze_eyeball_radius,
                    camera_pose=renderer.camera_pose,
                    yfov=renderer.camera.yfov,
                    aspect_ratio=renderer.camera.aspectRatio,
                    pupil_gain=pupil_gain,
                )

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
    parser = argparse.ArgumentParser(description='Render FLAME .npz to video with gaze eyeball overlay')
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
    parser.add_argument('--gaze-eyeball-radius', type=int, default=26,
                        help='Eyeball radius in pixels (default: 26)')
    parser.add_argument('--pupil-gain', type=float, default=1.8,
                        help='Gain for pupil motion inside eye (default: 1.8)')
    parser.add_argument('--no-gaze-eyeball', '--no-gaze-arrow',
                        dest='no_gaze_eyeball', action='store_true',
                        help='Disable gaze eyeball overlay even if gaze data exists')
    args = parser.parse_args()

    _ensure_project_root(args.fer_root)

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
        gaze_eyeball_radius=args.gaze_eyeball_radius,
        draw_gaze_eyeball_overlay=not args.no_gaze_eyeball,
        output_role_suffix=output_role_suffix,
        pupil_gain=args.pupil_gain,
    )


if __name__ == '__main__':
    main()