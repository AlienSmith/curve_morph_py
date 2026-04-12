import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from pipeline.pipeline import generate_morph  # Your path
matplotlib.use('Agg')

# ==============================
# CONFIGURATION
# ==============================
TARGET_SEGMENTS = 64  # 64 quad Bézier curves -> 128 points (A,C,A,C...)
# Try 32, 64, 96, or 128. Must be a multiple of 16 for clean subdivision.

# ==============================
# BÉZIER EVALUATION & RESAMPLING
# ==============================


def get_bezier(p0, p1, p2, steps=12):
    t = np.linspace(0, 1, steps)[:, None]
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2


def bezier_line(pts, steps=12):
    out = []
    n = len(pts)
    for i in range(n):
        out.append(get_bezier(pts[i], pts[(i+1) % n], pts[(i+2) % n], steps))
    return np.vstack(out)


def resample_by_arc_length(pts, n_out):
    pts_closed = np.vstack([pts, pts[0]])
    dists = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total_len = cumulative[-1]
    target_dists = np.linspace(0, total_len, n_out, endpoint=False)
    new_x = np.interp(target_dists, cumulative, pts_closed[:, 0])
    new_y = np.interp(target_dists, cumulative, pts_closed[:, 1])
    return np.c_[new_x, new_y]

# ==============================
# RESOLUTION UPGRADE (Core Feature)
# ==============================


def evaluate_high_res(editor_pts, samples_per_curve=128):
    """Evaluate original 32-pt curve at high resolution for accurate arc-length mapping."""
    pts = []
    n_curves = len(editor_pts) // 2
    for i in range(n_curves):
        p0 = editor_pts[2*i]
        p1 = editor_pts[2*i+1]
        p2 = editor_pts[2*(i+1) % len(editor_pts)]
        for j in range(samples_per_curve):
            t = j / samples_per_curve
            pts.append((1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2)
    return np.array(pts, dtype=np.float32)


def upgrade_to_uniform_resolution(editor_pts, target_segments=64):
    """
    1. Samples anchors uniformly by arc-length
    2. Reconstructs control points using exact C = 2*B(0.5) - 0.5*(A0 + A1)
    3. Returns [A0, C0, A1, C1, ...] of length 2*target_segments
    Preserves original shape exactly while increasing resolution for physics/FFT.
    """
    high_res = evaluate_high_res(editor_pts, samples_per_curve=256)

    # 1. Uniform arc-length anchors
    anchors = resample_by_arc_length(high_res, target_segments)

    # 2. Reconstruct controls via midpoints on original curve
    N = target_segments
    controls = np.zeros((N, 2), dtype=np.float32)

    cumulative = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(high_res, axis=0), axis=1))])
    total_len = cumulative[-1]

    for i in range(N):
        s_i = (i / N) * total_len
        s_next = ((i + 1) % N / N) * total_len
        if i == N - 1:
            s_next = total_len

        s_mid = (s_i + s_next) / 2.0

        # Interpolate midpoint from high-res curve
        mid_x = np.interp(s_mid, cumulative, high_res[:, 0])
        mid_y = np.interp(s_mid, cumulative, high_res[:, 1])
        mid_pt = np.array([mid_x, mid_y], dtype=np.float32)

        # Exact quadratic control reconstruction: C = 2*M - 0.5*(A0 + A1)
        A0 = anchors[i]
        A1 = anchors[(i+1) % N]
        controls[i] = 2.0 * mid_pt - 0.5 * (A0 + A1)

    # Interleave [A0, C0, A1, C1, ...]
    out = np.zeros((2*N, 2), dtype=np.float32)
    out[0::2] = anchors
    out[1::2] = controls
    return out

# ==============================
# JSON LOADER & PREPROCESSING
# ==============================


def load_and_upgrade(json_path, target_segments=64):
    with open(json_path, 'r') as f:
        editor_pts = np.array(json.load(f), dtype=np.float32)
    assert editor_pts.shape == (
        32, 2), f"Editor JSON must be (32,2), got {editor_pts.shape}"

    # Upgrade to higher resolution [A,C,A,C...]
    pts = upgrade_to_uniform_resolution(editor_pts, target_segments)

    # Enforce CCW winding
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + \
        0.5*(x[-1]*y[0] - x[0]*y[-1])
    if area < 0:
        pts = pts[::-1]
        pts = np.roll(pts, 1, axis=0)
    return pts

# ================================
# The loader with metadata support
# ================================


def load_and_upgrade_v1(json_path, target_segments=64):
    with open(json_path, 'r') as f:
        content = json.load(f)

    # --- SMART PARSING ---
    # Check if we have the new "Manifest" format or the old "Raw Array"
    if isinstance(content, dict) and "points" in content:
        editor_pts = np.array(content["points"], dtype=np.float32)
        meta = content.get("meta", {})
        winding_hint = meta.get("winding", "UNKNOWN")
        is_normalized = meta.get("normalized", False)
        print(
            f"📦 Loaded Manifest: {winding_hint}, Normalized: {is_normalized}")
    else:
        # Fallback for old 32x2 array files
        editor_pts = np.array(content, dtype=np.float32)
        winding_hint = "UNKNOWN"

    assert editor_pts.shape == (
        32, 2), f"Expected (32,2), got {editor_pts.shape}"

    # 1. Upgrade to higher resolution [A,C,A,C...]
    pts = upgrade_to_uniform_resolution(editor_pts, target_segments)

    # 2. ENFORCE CCW WINDING (Using the meta hint or manual calculation)
    x, y = pts[:, 0], pts[:, 1]
    # Shoelace formula for area
    area = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]
                         ) + (x[-1] * y[0] - x[0] * y[-1]))

    if area < 0:
        # Shape is Clockwise, flip it to CCW for the FFT
        pts = pts[::-1]
        # After flipping, we roll by 1 to maintain the [A, C, A, C] alignment
        pts = np.roll(pts, 1, axis=0)
        print("🔄 Fixed Winding: Flipped CW to CCW")

    return pts


def run_morph_test_hard():
    TARGET_SEGMENTS = 128
    print("🔹 Loading shapes...")
    A = load_and_upgrade_v1("/workspace/C_shape.json", TARGET_SEGMENTS)
    B = load_and_upgrade_v1("/workspace/circle.json", TARGET_SEGMENTS)

    print("🔹 Running PBD (Distance + Fourier only)...")
    frames, boundary_idx = generate_morph(A, B, num_frames=61)

    print("🔹 Rendering GIF...")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Plot as SCATTER first to verify positions (no ordering assumptions)
    sc_boundary = ax.scatter([], [], s=8, c='steelblue', zorder=3)
    sc_interior = ax.scatter([], [], s=4, c='lightgray', alpha=0.5, zorder=2)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        all_pts = frames[f]
        b_pts = all_pts[boundary_idx]
        i_pts = np.delete(all_pts, boundary_idx, axis=0)

        sc_boundary.set_offsets(b_pts)
        sc_interior.set_offsets(i_pts)
        title.set_text(f"Frame {f} | {len(b_pts)} boundary pts")
        return sc_boundary, sc_interior, title

    anim = FuncAnimation(fig, update, frames=61, blit=False, interval=33)
    anim.save("morph_debug.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_debug.gif")


def make_shape(cx, cy, r1, r2, n=128, phase=0.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
    return np.stack([cx + r1*np.cos(t), cy + r2*np.sin(t)], axis=1)


def run_morph_test():
    print("🔹 Generating test shapes...")
    A = make_shape(0.0, 0.0, 1.0, 0.6)  # Ellipse
    B = make_shape(0.2, 0.1, 0.8, 0.8, phase=np.pi/4)  # Rotated circle

    print("🔹 Running PBD pipeline...")
    frames, boundary_idx = generate_morph(A, B, num_frames=40)

    print("🔹 Rendering GIF...")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    sc_boundary = ax.scatter([], [], s=10, c='steelblue', zorder=3)
    sc_interior = ax.scatter([], [], s=4, c='lightgray', alpha=0.4, zorder=2)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=11)

    def update(f):
        all_pts = frames[f]
        sc_boundary.set_offsets(all_pts[boundary_idx])
        sc_interior.set_offsets(np.delete(all_pts, boundary_idx, axis=0))
        title.set_text(f"Frame {f} | {len(boundary_idx)} boundary pts")
        return sc_boundary, sc_interior, title

    FuncAnimation(fig, update, frames=len(frames), blit=False, interval=30).save(
        "pbd_morph_test.gif", fps=24, writer='pillow')
    print("✅ Saved: pbd_morph_test.gif")


if __name__ == "__main__":
    run_morph_test_hard()
