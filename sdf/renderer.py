# sdf_renderer.py
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter, gaussian_filter
from skimage.draw import polygon


class SDFRenderer:
    def __init__(self, pts_a: np.ndarray, pts_b: np.ndarray, grid_size: int = 512, padding_ratio: float = 0.15):
        self.grid_size = grid_size
        self.padding_ratio = padding_ratio

        self.curve_a = self._sample_curve_dense(pts_a)
        self.curve_b = self._sample_curve_dense(pts_b)

        # 1. Union bounding box
        all_pts = np.vstack([self.curve_a, self.curve_b])
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)

        half_w = 0.5 * (x_max - x_min)
        half_h = 0.5 * (y_max - y_min)
        half_dim = max(half_w, half_h) * (1.0 + padding_ratio)

        self.x_min, self.x_max = cx - half_dim, cx + half_dim
        self.y_min, self.y_max = cy - half_dim, cy + half_dim
        self.pixel_scale = (self.x_max - self.x_min) / self.grid_size

        # ✅ PRECOMPUTE SDFs
        self.sdf_a = self._compute_signed_sdf(self._rasterize(self.curve_a))
        self.sdf_b = self._compute_signed_sdf(self._rasterize(self.curve_b))

    def _sample_curve_dense(self, pts: np.ndarray, samples_per_seg: int = 64) -> np.ndarray:
        N = len(pts) // 2
        t = np.linspace(0, 1, samples_per_seg, endpoint=False)
        t2 = t[:, np.newaxis]
        P0 = pts[0::2][np.newaxis, :, :]
        P1 = pts[1::2][np.newaxis, :, :]
        P2 = np.roll(pts[0::2][np.newaxis, :, :], -1, axis=1)
        curve = (1 - t2)**2 * P0 + 2 * (1 - t2) * t2 * P1 + t2**2 * P2
        return curve.reshape(-1, 2)

    def _rasterize(self, pts: np.ndarray) -> np.ndarray:
        px = ((pts[:, 0] - self.x_min) / self.pixel_scale).astype(int)
        py = ((pts[:, 1] - self.y_min) / self.pixel_scale).astype(int)
        px = np.clip(px, 0, self.grid_size - 1)
        py = np.clip(py, 0, self.grid_size - 1)
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        rr, cc = polygon(py, px)
        mask[rr, cc] = 1.0
        return mask

    def _compute_signed_sdf(self, mask: np.ndarray) -> np.ndarray:
        dist_out = distance_transform_edt(mask == 0)
        dist_in = distance_transform_edt(mask > 0)
        return (dist_out - dist_in) * self.pixel_scale

    def render(self, alpha: float, bias: float = 0.0) -> np.ndarray:
        """
        Interpolated SDF with pinch-targeted topology stabilization.
        Only biases regions where |SDF| is small AND local variance is low (flat bottleneck).
        Leaves corners, steep slopes, and deep interiors untouched.
        """
        sdf = (1.0 - alpha) * self.sdf_a + alpha * self.sdf_b

        if bias > 1e-6:
            # 1. Restrict to narrow band around zero-crossing
            near_zero = np.abs(sdf) < 2.0  # world units

            # 2. Compute local variance in a sliding window (gradient-free)
            # var(X) = E[X²] - E[X]²
            win_size = 9  # ~9-pixel window, covers typical pinch width
            mean_sdf = uniform_filter(sdf, size=win_size)
            mean_sdf2 = uniform_filter(sdf**2, size=win_size)
            variance = np.maximum(mean_sdf2 - mean_sdf**2, 0.0)

            # 3. Soft pinch mask: high where variance is low, zero elsewhere
            # 0.3 is the "flatness threshold" (tune if needed)
            pinch_mask = np.exp(-variance / 0.3) * near_zero

            # 4. Smooth mask edges to avoid blocky artifacts
            pinch_mask = gaussian_filter(pinch_mask, sigma=1.5)

            # 5. Apply localized bias
            sdf += bias * pinch_mask

        return sdf

    def get_grid_metadata(self) -> dict:
        return {
            "x_min": self.x_min, "x_max": self.x_max,
            "y_min": self.y_min, "y_max": self.y_max,
            "pixel_scale": self.pixel_scale, "grid_size": self.grid_size
        }
