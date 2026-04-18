import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# ==========================================================
# CORE UTILITIES
# ==========================================================


def resample_by_arc_length(pts: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a closed polygon to n_out equidistant points by arc length."""
    pts_closed = np.vstack([pts, pts[0]])
    dists = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total_len = cumulative[-1]
    target_dists = np.linspace(0, total_len, n_out, endpoint=False)
    new_x = np.interp(target_dists, cumulative, pts_closed[:, 0])
    new_y = np.interp(target_dists, cumulative, pts_closed[:, 1])
    return np.c_[new_x, new_y]


def evaluate_quadratic_bezier(high_res_pts: int = 256) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a function that evaluates a sequence of quad Béziers at high resolution."""
    def _eval(control_points: np.ndarray) -> np.ndarray:
        # control_points: [A0, C0, A1, C1, ...] of shape (2N, 2)
        N = len(control_points) // 2
        pts = []
        for i in range(N):
            p0 = control_points[2*i]
            p1 = control_points[2*i+1]
            p2 = control_points[2*(i+1) % (2*N)]  # wraps to next anchor
            for j in range(high_res_pts):
                t = j / high_res_pts
                pts.append((1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2)
        return np.array(pts, dtype=np.float32)
    return _eval


def upgrade_to_uniform_resolution(editor_pts: np.ndarray, target_segments: int) -> np.ndarray:
    """Upgrade quad Bézier control points to target_segments while preserving shape."""
    high_res_fn = evaluate_quadratic_bezier(high_res_pts=256)
    high_res = high_res_fn(editor_pts)

    # 1. Uniform arc-length anchors
    anchors = resample_by_arc_length(high_res, target_segments)

    N = target_segments
    controls = np.zeros((N, 2), dtype=np.float32)
    cumulative = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(high_res, axis=0), axis=1))])
    total_len = cumulative[-1]

    # 2. Reconstruct controls via midpoints on original curve
    for i in range(N):
        s_mid = ((i + 0.5) / N) * total_len
        mid_x = np.interp(s_mid, cumulative, high_res[:, 0])
        mid_y = np.interp(s_mid, cumulative, high_res[:, 1])
        mid_pt = np.array([mid_x, mid_y], dtype=np.float32)

        A0 = anchors[i]
        A1 = anchors[(i + 1) % N]
        # Exact quadratic control reconstruction: C = 2*M - 0.5*(A0 + A1)
        controls[i] = 2.0 * mid_pt - 0.5 * (A0 + A1)

    # Interleave [A0, C0, A1, C1, ...]
    out = np.zeros((2 * N, 2), dtype=np.float32)
    out[0::2] = anchors
    out[1::2] = controls
    return out


def enforce_ccw_winding(pts: np.ndarray) -> np.ndarray:
    """Ensure anchor points are CCW. pts is expected in [A0, C0, A1, C1, ...] format."""
    anchors = pts[0::2]  # Only use anchor points for area calculation
    x, y = anchors[:, 0], anchors[:, 1]
    area = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]
                         ) + (x[-1] * y[0] - x[0] * y[-1]))

    if area < 0:
        pts = pts[::-1]
        # Maintain [A,C,A,C] alignment after flip
        pts = np.roll(pts, 1, axis=0)
    return pts


def load_and_upgrade(json_path: str, target_segments: int) -> np.ndarray:
    with open(json_path, 'r') as f:
        content = json.load(f)

    if isinstance(content, dict) and "points" in content:
        editor_pts = np.array(content["points"], dtype=np.float32)
    else:
        editor_pts = np.array(content, dtype=np.float32)

    assert editor_pts.shape[1] == 2, f"Expected (N, 2), got {editor_pts.shape}"

    pts = upgrade_to_uniform_resolution(editor_pts, target_segments)
    pts = enforce_ccw_winding(pts)
    return pts


# ==========================================================
# TEST & VISUALIZATION
# ==========================================================

def generate_test_control_points(n_segments: int = 8) -> np.ndarray:
    """Generate a simple circle-like quad Bézier chain for testing."""
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    radius = 1.0
    anchors = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    controls = np.zeros((n_segments, 2))

    for i in range(n_segments):
        p0 = anchors[i]
        p1 = anchors[(i+1) % n_segments]
        mid = 0.5 * (p0 + p1)
        edge = p1 - p0
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Push control outward to approximate a circular arc
        controls[i] = mid + normal * 0.15

    out = np.zeros((2 * n_segments, 2), dtype=np.float32)
    out[0::2] = anchors
    out[1::2] = controls
    return out


if __name__ == "__main__":
    # 1. Generate synthetic input [A, C, A, C, ...]
    raw_pts = generate_test_control_points(n_segments=8)
    print(f"📥 Raw input shape: {raw_pts.shape} (anchors + controls)")

    # 2. Upgrade to higher resolution
    target_segs = 16
    upgraded_pts = upgrade_to_uniform_resolution(raw_pts, target_segs)
    print(
        f"⬆️ Upgraded shape: {upgraded_pts.shape} (target_segments={target_segs})")

    # 3. Enforce CCW winding & verify
    final_pts = enforce_ccw_winding(upgraded_pts)
    anchors = final_pts[0::2]
    x, y = anchors[:, 0], anchors[:, 1]
    area = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]
                         ) + (x[-1] * y[0] - x[0] * y[-1]))
    print(f"🔍 Winding area: {area:.4f} {'(CCW ✅)' if area > 0 else '(CW ❌)'}")

    # 4. Evaluate high-res curves for plotting
    eval_fn = evaluate_quadratic_bezier(high_res_pts=128)
    raw_curve = eval_fn(raw_pts)
    final_curve = eval_fn(final_pts)

    # 5. Plot results
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Original curve (dashed)
    ax.plot(raw_curve[:, 0], raw_curve[:, 1], 'k--',
            linewidth=1.5, alpha=0.4, label='Original Curve')

    # Upgraded curve (solid)
    ax.plot(final_curve[:, 0], final_curve[:, 1],
            'b-', linewidth=2, label='Upgraded Curve')

    # Anchors & Controls
    final_anchors = final_pts[0::2]
    final_controls = final_pts[1::2]
    ax.plot(final_anchors[:, 0], final_anchors[:, 1],
            'go', markersize=5, label='Anchors')
    ax.plot(final_controls[:, 0], final_controls[:, 1],
            'rx', markersize=6, label='Controls')

    # Draw control polygon lines for clarity
    for i in range(target_segs):
        A0 = final_anchors[i]
        C = final_controls[i]
        A1 = final_anchors[(i+1) % target_segs]
        ax.plot([A0[0], C[0], A1[0]], [A0[1], C[1], A1[1]],
                'r-', linewidth=0.8, alpha=0.3)

    ax.legend(loc='upper right')
    ax.set_title("Quad Bézier Upgrade & Winding Verification")
    plt.tight_layout()
    plt.show()

    # Optional: Replace with your actual JSON files to test real pipeline
    # A = load_and_upgrade("/workspace/C_shape.json", target_segs)
    # B = load_and_upgrade("/workspace/circle.json", target_segs)
