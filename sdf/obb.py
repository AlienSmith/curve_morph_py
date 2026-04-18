# obb_align.py
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class OBB2:
    center: np.ndarray
    axis_x: np.ndarray
    axis_y: np.ndarray
    half_size: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray) -> Optional['OBB2']:
        if len(points) == 0:
            return None
        if len(points) == 1:
            return cls(
                center=points[0],
                axis_x=np.array([1.0, 0.0], dtype=np.float32),
                axis_y=np.array([0.0, 1.0], dtype=np.float32),
                half_size=np.zeros(2, dtype=np.float32)
            )

        # 1. Lexicographic sort & deduplicate
        pts = points[np.lexsort((points[:, 1], points[:, 0]))]
        pts = pts[np.concatenate(
            ([True], np.any(np.diff(pts, axis=0), axis=1)))]

        if len(pts) <= 2:
            return cls(
                center=np.mean(pts, axis=0),
                axis_x=np.array([1.0, 0.0], dtype=np.float32),
                axis_y=np.array([0.0, 1.0], dtype=np.float32),
                half_size=np.zeros(2, dtype=np.float32)
            )

        # 2. Monotone Chain Convex Hull
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        lower, upper = [], []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)

        # 3. Rotating Calipers (Vectorized Projection)
        min_area = np.inf
        best_obb = None
        n_hull = len(hull)

        for i in range(n_hull):
            p1, p2 = hull[i], hull[(i + 1) % n_hull]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue

            axis_x = edge / edge_len
            axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float32)

            # Project all hull points onto local axes (fully vectorized)
            proj_x = hull @ axis_x
            proj_y = hull @ axis_y

            width = proj_x.max() - proj_x.min()
            height = proj_y.max() - proj_y.min()
            area = width * height

            if area < min_area:
                min_area = area
                center_u = (proj_x.max() + proj_x.min()) * 0.5
                center_v = (proj_y.max() + proj_y.min()) * 0.5
                center = axis_x * center_u + axis_y * center_v

                best_obb = cls(
                    center=center,
                    axis_x=axis_x,
                    axis_y=axis_y,
                    half_size=np.array(
                        [width * 0.5, height * 0.5], dtype=np.float32)
                )
        return best_obb


class OBBAligner:
    """Computes tight OBB alignment & provides canonical/world transforms."""

    def __init__(self, pts_a: np.ndarray, pts_b: np.ndarray):
        # OBB only cares about the actual boundary (anchors)
        self.obb_a = OBB2.from_points(pts_a[0::2])
        self.obb_b = OBB2.from_points(pts_b[0::2])

        # Precompute World->Canonical rotation matrices
        # Rows are the local basis vectors
        self._R_a_wc = np.vstack([self.obb_a.axis_x, self.obb_a.axis_y])
        self._R_b_wc = np.vstack([self.obb_b.axis_x, self.obb_b.axis_y])

    def to_canonical(self, pts: np.ndarray, shape: str = 'a') -> np.ndarray:
        """Transform world points to canonical space (centered, major axis +X)."""
        R = self._R_a_wc if shape == 'a' else self._R_b_wc
        c = self.obb_a.center if shape == 'a' else self.obb_b.center
        return (pts - c) @ R.T

    def to_world(self, alpha: float, pts_canonical: np.ndarray) -> np.ndarray:
        """Interpolate alignment transforms & map canonical points back to world."""
        # Linear blend of centers
        center = (1 - alpha) * self.obb_a.center + alpha * self.obb_b.center

        # Robust 2D angle interpolation (avoids ±π jumps)
        ang_a = np.arctan2(self.obb_a.axis_x[1], self.obb_a.axis_x[0])
        ang_b = np.arctan2(self.obb_b.axis_x[1], self.obb_b.axis_x[0])

        cos_interp = (1 - alpha) * np.cos(ang_a) + alpha * np.cos(ang_b)
        sin_interp = (1 - alpha) * np.sin(ang_a) + alpha * np.sin(ang_b)
        angle = np.arctan2(sin_interp, cos_interp)

        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

        # Canonical -> World
        return (pts_canonical @ R.T) + center
