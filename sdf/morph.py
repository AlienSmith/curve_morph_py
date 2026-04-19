import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from loader import load_and_upgrade
from obb import OBBAligner
from renderer import SDFRenderer
from sampler import ContourSampler
from utils import evaluate_bezier_curve
from typing import Dict, Tuple


def _compute_contour_curvature(pts: np.ndarray) -> np.ndarray:
    """Compute absolute curvature magnitude for equidistant closed contour points."""
    if len(pts) < 3:
        return np.zeros(len(pts))

    pts_closed = np.vstack([pts, pts[0]])
    p_prev, p_curr, p_next = pts_closed[:-2], pts_closed[1:-1], pts_closed[2:]

    ds = np.linalg.norm(p_next - p_curr, axis=1).mean()
    if ds < 1e-8:
        return np.zeros(len(p_curr))

    return np.linalg.norm(p_next - 2 * p_curr + p_prev, axis=1) / (ds**2)


def _align_phase_and_compute_error(c_a: np.ndarray, c_b: np.ndarray, c_mid: np.ndarray) -> Tuple[float, float]:
    """
    Finds the optimal circular shift of c_mid to align with c_a,
    then computes positional and curvature errors relative to linear interpolation.
    Phase-invariant: immune to sampling start-point jumps.
    """
    N = len(c_a)
    c_lin = (c_a + c_b) / 2.0

    best_mse = np.inf
    best_shifted_mid = c_mid
    for shift in range(N):
        c_s = np.roll(c_mid, shift, axis=0)
        mse = np.mean(np.linalg.norm(c_s - c_lin, axis=1)**2)
        if mse < best_mse:
            best_mse = mse
            best_shifted_mid = c_s

    pos_err = np.sqrt(best_mse)

    k_a = _compute_contour_curvature(c_a)
    k_b = _compute_contour_curvature(c_b)
    k_mid = _compute_contour_curvature(best_shifted_mid)
    linear_k = (k_a + k_b) / 2.0
    k_err = float(np.max(np.abs(k_mid - linear_k)))

    return pos_err, k_err


def _get_adaptive_bias(t: float, bbox_diag: float) -> float:
    """
    Smoothly lifts narrow necks above SDF=0 to prevent topology splits.
    Strictly 0.0 at t=0 and t=1 to guarantee exact endpoint shapes.
    """
    if t == 0.0 or t == 1.0:
        return 0.0
    if t <= 0.001 or t >= 0.999:
        return 0.0
    # Scales to ~2% of shape extent; peaks at t=0.5
    return 0.02 * bbox_diag * np.sin(np.pi * t)


def generate_morph_sequence(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    target_segments: int,
    error_threshold: float = 0.0015,
    min_t_step: float = 0.008,
    max_frames: int = 200,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive morph sequence generator with topology-stable SDF blending.
    Guarantees exact start/end shapes (zero bias at endpoints).
    """
    aligner = OBBAligner(pts_a, pts_b)
    A_can = aligner.to_canonical(pts_a, 'a')
    B_can = aligner.to_canonical(pts_b, 'b')

    # ✅ Compute bbox_diag BEFORE get_contour is defined/called
    all_can_pts = np.vstack([A_can, B_can])
    bbox_diag = float(np.linalg.norm(
        all_can_pts.max(axis=0) - all_can_pts.min(axis=0)))
    if bbox_diag < 1e-6:
        bbox_diag = 1.0

    renderer = SDFRenderer(A_can, B_can, grid_size=512, padding_ratio=0.15)
    metadata = renderer.get_grid_metadata()
    n_pts = target_segments * 2

    contour_cache: Dict[float, np.ndarray] = {}
    sdf_cache: Dict[float, np.ndarray] = {}

    def get_contour(t: float) -> np.ndarray:
        t_key = round(float(t), 8)
        if t_key not in contour_cache:
            bias = _get_adaptive_bias(t, bbox_diag)
            sdf = renderer.render(t, bias=bias)
            sdf_cache[t_key] = sdf
            contour_cache[t_key] = ContourSampler(
                sdf, metadata).sample_boundary(min(n_pts, 96))
        return contour_cache[t_key]

    get_contour(0.0)
    get_contour(1.0)

    t_vals = [0.0, 1.0]

    while len(t_vals) < max_frames:
        worst_err = 0.0
        worst_idx = -1

        for i in range(len(t_vals) - 1):
            t_a, t_b = float(t_vals[i]), float(t_vals[i + 1])
            if t_b - t_a < min_t_step:
                continue

            t_mid = float((t_a + t_b) / 2.0)
            c_a = contour_cache[round(t_a, 8)]
            c_b = contour_cache[round(t_b, 8)]
            c_mid = get_contour(t_mid)

            pos_err, k_err = _align_phase_and_compute_error(c_a, c_b, c_mid)

            bbox_diag_local = np.linalg.norm(c_a.max(axis=0) - c_a.min(axis=0))
            scale = max(bbox_diag_local, 1e-6)
            err_norm = (0.7 * pos_err + 0.3 * k_err) / scale

            if err_norm > worst_err:
                worst_err = err_norm
                worst_idx = i

        if verbose:
            print(
                f"  Frames: {len(t_vals):2d} | worst_err={worst_err:.4f} | threshold={error_threshold}")

        if worst_idx == -1 or worst_err <= error_threshold:
            break

        t_a, t_b = t_vals[worst_idx], t_vals[worst_idx + 1]
        t_mid = float((t_a + t_b) / 2.0)
        t_vals.insert(worst_idx + 1, t_mid)

    # Final sequence generation
    t_vals = sorted(list(set([round(float(t), 8) for t in t_vals])))
    sequence = np.zeros((len(t_vals), n_pts, 2), dtype=np.float32)

    for i, alpha in enumerate(t_vals):
        alpha_key = round(float(alpha), 8)

        # ✅ GUARANTEE EXACT ENDPOINTS: bypass bias entirely at t=0,1
        if alpha_key == 0.0:
            sdf = renderer.sdf_a
        elif alpha_key == 1.0:
            sdf = renderer.sdf_b
        else:
            if alpha_key in sdf_cache:
                sdf = sdf_cache[alpha_key]
            else:
                bias = _get_adaptive_bias(alpha, bbox_diag)
                sdf = renderer.render(alpha, bias=bias)

        canonical_pts = ContourSampler(sdf, metadata).sample_boundary(n_pts)
        world_pts = aligner.to_world(alpha, canonical_pts)

        # Interleave: even indices → Controls, odd indices → Anchors
        sequence[i, 0::2] = world_pts[1::2]
        sequence[i, 1::2] = world_pts[0::2]

    return sequence, np.array(t_vals, dtype=np.float32)


if __name__ == "__main__":
    TARGET_SEGMENTS = 64
    GIF_PATH = "/workspace/morph_preview.gif"
    JSON_PATH = "/workspace/morph_output.json"

    print("📥 Loading & upgrading shapes...")
    A = load_and_upgrade("/workspace/C_shape.json", TARGET_SEGMENTS)
    B = load_and_upgrade("/workspace/circle.json", TARGET_SEGMENTS)

    print("⚙️ Generating adaptive morph sequence...")
    morph_sequence, t_values = generate_morph_sequence(A, B, TARGET_SEGMENTS)
    num_frames = morph_sequence.shape[0]
    print(
        f"✅ Output shape: {morph_sequence.shape} | Frames generated: {num_frames}")

    with open(JSON_PATH, 'w') as f:
        json.dump({
            "points": morph_sequence.tolist(),
            "t_values": t_values.tolist(),
            "meta": {
                "target_segments": TARGET_SEGMENTS,
                "num_frames": num_frames,
                "format": "[Control, Anchor, Control, Anchor, ...] per frame"
            }
        }, f, indent=2)
    print(f"💾 Saved JSON to {JSON_PATH}")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_aspect('equal')
    ax.axis('off')

    all_pts = morph_sequence.reshape(-1, 2)
    pad = 0.4
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    curve_line, = ax.plot([], [], 'b-', lw=2.2)
    anchor_dots, = ax.plot([], [], 'ro', ms=4, alpha=0.7)
    title_txt = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                        ha='center', fontsize=11)

    def update(f):
        pts = morph_sequence[f]
        curve = evaluate_bezier_curve(pts, samples_per_seg=64)
        curve_line.set_data(curve[:, 0], curve[:, 1])
        anchor_dots.set_data(pts[1::2, 0], pts[1::2, 1])
        title_txt.set_text(
            f"Frame {f}/{num_frames - 1} | α = {t_values[f]:.3f}")
        return curve_line, anchor_dots, title_txt

    print("🎬 Rendering GIF...")
    anim = FuncAnimation(fig, update, frames=num_frames,
                         blit=False, interval=100)
    anim.save(GIF_PATH, fps=5, writer='pillow')
    print(f"✅ Saved GIF to {GIF_PATH}")

    plt.close(fig)
