import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from pipeline.pipeline import generate_morph
matplotlib.use('Agg')

# ==============================
# CONFIGURATION
# ==============================
TARGET_SEGMENTS = 64

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
    high_res = evaluate_high_res(editor_pts, samples_per_curve=256)
    anchors = resample_by_arc_length(high_res, target_segments)
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
        mid_x = np.interp(s_mid, cumulative, high_res[:, 0])
        mid_y = np.interp(s_mid, cumulative, high_res[:, 1])
        mid_pt = np.array([mid_x, mid_y], dtype=np.float32)
        A0 = anchors[i]
        A1 = anchors[(i+1) % N]
        controls[i] = 2.0 * mid_pt - 0.5 * (A0 + A1)

    out = np.zeros((2*N, 2), dtype=np.float32)
    out[0::2] = anchors
    out[1::2] = controls
    return out
# ================================
# The loader with metadata support
# ================================


def load_and_upgrade_v1(json_path, target_segments=64):
    with open(json_path, 'r') as f:
        content = json.load(f)

    if isinstance(content, dict) and "points" in content:
        editor_pts = np.array(content["points"], dtype=np.float32)
        meta = content.get("meta", {})
        winding_hint = meta.get("winding", "UNKNOWN")
        is_normalized = meta.get("normalized", False)
        print(
            f"📦 Loaded Manifest: {winding_hint}, Normalized: {is_normalized}")
    else:
        editor_pts = np.array(content, dtype=np.float32)
        winding_hint = "UNKNOWN"

    assert editor_pts.shape == (
        32, 2), f"Expected (32,2), got {editor_pts.shape}"
    pts = upgrade_to_uniform_resolution(editor_pts, target_segments)

    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]
                         ) + (x[-1] * y[0] - x[0] * y[-1]))
    if area < 0:
        pts = pts[::-1]
        pts = np.roll(pts, 1, axis=0)
        print("🔄 Fixed Winding: Flipped CW to CCW")

    return pts

# ===========================================================================
# 👇 ONLY THIS FUNCTION IS MODIFIED — uses get_bezier for SMOOTH BOUNDARY
# ===========================================================================


def run_morph_test_hard():
    TARGET_SEGMENTS = 128
    print("🔹 Loading shapes...")
    # A = load_and_upgrade_v1("/workspace/C_shape.json", TARGET_SEGMENTS)
    # B = load_and_upgrade_v1("/workspace/circle.json", TARGET_SEGMENTS)
    A = make_shape(0.0, 0.0, 1.0, 0.6)
    B = make_shape(0.2, 0.1, 0.8, 0.8, phase=np.pi/4)

    print("🔹 Running PBD (Distance + Fourier only)...")
    frames, boundary_idx = generate_morph(A, B, num_frames=61)

    print("🔹 Rendering GIF with smooth Bézier boundary...")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # --------------------------
    # SMOOTH BÉZIER CURVE (using YOUR get_bezier)
    # --------------------------
    bezier_line_plot, = ax.plot([], [], lw=2, c='steelblue', zorder=3)
    sc_interior = ax.scatter([], [], s=4, c='lightgray', alpha=0.5, zorder=2)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        all_pts = frames[f]
        b_pts = all_pts[boundary_idx]  # Raw boundary vertices
        i_pts = np.delete(all_pts, boundary_idx, axis=0)

        # --------------------------
        # USE YOUR get_bezier TO DRAW SMOOTH CURVE
        # --------------------------
        smooth_curve = bezier_line(b_pts, steps=8)  # steps=8 → smooth but fast
        bezier_line_plot.set_data(smooth_curve[:, 0], smooth_curve[:, 1])
        sc_interior.set_offsets(i_pts)
        title.set_text(f"Frame {f} | Smooth Bézier Boundary")

        return bezier_line_plot, sc_interior, title

    anim = FuncAnimation(fig, update, frames=61, blit=False, interval=33)
    anim.save("morph_bezier.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_bezier.gif (smooth Bézier boundary!)")


def make_shape(cx, cy, r1, r2, n=128, phase=0.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
    return np.stack([cx + r1*np.cos(t), cy + r2*np.sin(t)], axis=1)


if __name__ == "__main__":
    run_morph_test_hard()
