import numpy as np


def reconstruct_bezier_controls(pts: np.ndarray, anchors_at_odd_indices: bool = False) -> np.ndarray:
    N = len(pts)
    assert N % 2 == 0, "Boundary points must be even for alternating A/C interleaving."
    anchors = pts[1::2] if anchors_at_odd_indices else pts[0::2]
    controls = pts[0::2] if anchors_at_odd_indices else pts[1::2]
    out = np.empty((N, 2), dtype=np.float32)
    out[0::2] = anchors
    out[1::2] = controls
    return out


def evaluate_bezier_curve(interleaved_pts: np.ndarray, samples_per_seg: int = 48) -> np.ndarray:
    """Densely evaluates a closed quad Bézier chain from [C, A, C, A, ...] format."""
    # Reorder to [A, C, A, C, ...] for standard quadratic evaluation
    pts = np.empty_like(interleaved_pts)
    pts[0::2] = interleaved_pts[1::2]  # Anchors
    pts[1::2] = interleaved_pts[0::2]  # Controls

    N = len(pts) // 2
    t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, np.newaxis]

    P0 = pts[0::2][np.newaxis]
    P1 = pts[1::2][np.newaxis]
    P2 = np.roll(pts[0::2][np.newaxis], -1, axis=1)  # Wrap to next anchor

    # Quadratic Bézier: (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    curve = ((1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2).reshape(-1, 2)
    return curve
