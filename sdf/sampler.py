# contour_sampler.py
import numpy as np
from skimage.measure import find_contours
from typing import Dict


class ContourSampler:
    """
    Extracts the zero-level boundary from an SDF, aligns it to the +X axis,
    and resamples to a fixed number of equidistant points in canonical space.
    """

    def __init__(self, sdf: np.ndarray, metadata: Dict):
        self.sdf = sdf
        self.meta = metadata
        self.grid_size = metadata["grid_size"]
        self.pixel_scale = metadata["pixel_scale"]
        self.x_min = metadata["x_min"]
        self.y_min = metadata["y_min"]

    def _pixel_to_canonical(self, row_col: np.ndarray) -> np.ndarray:
        """Convert skimage (row, col) coordinates to canonical (x, y) world units."""
        col = row_col[:, 1]
        row = row_col[:, 0]
        # Row 0 in the SDF array corresponds to y_min (matches rasterizer logic)
        x = self.x_min + col * self.pixel_scale
        y = self.y_min + row * self.pixel_scale
        return np.c_[x, y]

    def _enforce_ccw(self, pts: np.ndarray) -> np.ndarray:
        """Flip contour to CCW if signed area is negative."""
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + \
            0.5*(x[-1]*y[0] - x[0]*y[-1])
        return pts if area > 0 else pts[::-1]

    def sample_boundary(self, n_points: int) -> np.ndarray:
        # 1. Extract sub-pixel accurate zero-level contour
        contours = find_contours(
            self.sdf, level=0.0, fully_connected='high', positive_orientation='low')
        if not contours:
            raise RuntimeError("No zero-level contour found in SDF.")

        # User guarantees single continuous boundary; take longest just in case
        contour_px = max(contours, key=len)

        # 2. Map to canonical coordinates & close loop
        contour = self._pixel_to_canonical(contour_px)
        contour = np.vstack([contour, contour[0]])  # Explicit closure

        # 3. Ensure CCW winding for consistent downstream processing
        contour = self._enforce_ccw(contour)

        # 4. Find start point: intersection closest to +X canonical axis (angle ≈ 0)
        angles = np.arctan2(contour[:, 1], contour[:, 0])
        start_idx = np.argmin(np.abs(angles))
        contour = np.roll(contour, -start_idx, axis=0)

        # 5. Arc-length resampling to exactly n_points
        return self._resample_arc_length(contour[:-1], n_points)

    def _resample_arc_length(self, pts: np.ndarray, n_out: int) -> np.ndarray:
        pts_closed = np.vstack([pts, pts[0]])
        dists = np.linalg.norm(np.diff(pts_closed, axis=0), axis=1)
        cumulative = np.concatenate([[0], np.cumsum(dists)])
        total_len = cumulative[-1]
        target_dists = np.linspace(0, total_len, n_out, endpoint=False)
        new_x = np.interp(target_dists, cumulative, pts_closed[:, 0])
        new_y = np.interp(target_dists, cumulative, pts_closed[:, 1])
        return np.c_[new_x, new_y]
