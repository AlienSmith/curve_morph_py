# sdf_renderer.py
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon
from typing import Dict


class SDFRenderer:
    """
    Rasterizes quad Bézier curves, computes Signed Distance Fields (SDF),
    and blends them for morphing. Outputs are in world units.
    """


class SDFRenderer:
    def __init__(self, pts_a: np.ndarray, pts_b: np.ndarray, grid_size: int = 512, padding_ratio: float = 0.15):
        self.grid_size = grid_size
        self.padding_ratio = padding_ratio

        # Dense sampling for accurate rasterization
        self.curve_a = self._sample_curve_dense(pts_a)
        self.curve_b = self._sample_curve_dense(pts_b)

        # 1. Tight union bounding box
        all_pts = np.vstack([self.curve_a, self.curve_b])
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)

        # 2. Center on the union centroid (usually ~0,0 after OBB alignment)
        cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)

        # 3. Square grid with relative padding
        half_w = 0.5 * (x_max - x_min)
        half_h = 0.5 * (y_max - y_min)
        half_dim = max(half_w, half_h) * (1.0 + padding_ratio)

        self.x_min, self.x_max = cx - half_dim, cx + half_dim
        self.y_min, self.y_max = cy - half_dim, cy + half_dim
        self.pixel_scale = (self.x_max - self.x_min) / self.grid_size

    def _sample_curve_dense(self, pts: np.ndarray, samples_per_seg: int = 64) -> np.ndarray:
        """Vectorized evaluation of quadratic Bézier segments."""
        N = len(pts) // 2
        t = np.linspace(0, 1, samples_per_seg, endpoint=False)
        t2 = t[:, np.newaxis]

        P0 = pts[0::2][np.newaxis, :, :]
        P1 = pts[1::2][np.newaxis, :, :]
        P2 = np.roll(pts[0::2][np.newaxis, :, :], -
                     1, axis=1)  # Wrap to next anchor

        curve = (1 - t2)**2 * P0 + 2 * (1 - t2) * t2 * P1 + t2**2 * P2
        return curve.reshape(-1, 2)

    def _rasterize(self, pts: np.ndarray) -> np.ndarray:
        """Convert curve points to a binary mask on the grid."""
        # World -> Pixel mapping
        px = ((pts[:, 0] - self.x_min) / self.pixel_scale).astype(int)
        py = ((pts[:, 1] - self.y_min) / self.pixel_scale).astype(int)

        # Clamp to grid bounds
        px = np.clip(px, 0, self.grid_size - 1)
        py = np.clip(py, 0, self.grid_size - 1)

        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        rr, cc = polygon(py, px)  # skimage uses (row, col) -> (y, x)
        mask[rr, cc] = 1.0
        return mask

    def _compute_signed_sdf(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute exact Euclidean SDF:
          - Negative inside shape
          - Positive outside shape
          - Zero on boundary
        Scales back to world units.
        """
        # Distance from outside pixels to boundary
        dist_out = distance_transform_edt(mask == 0)
        # Distance from inside pixels to boundary
        dist_in = distance_transform_edt(mask > 0)

        # Combine with sign convention
        sdf_pixels = dist_out - dist_in

        # Convert from pixels to world units
        return sdf_pixels * self.pixel_scale

    def render(self, alpha: float) -> np.ndarray:
        """Generate the interpolated SDF for a given morph parameter."""
        mask_a = self._rasterize(self.curve_a)
        mask_b = self._rasterize(self.curve_b)

        sdf_a = self._compute_signed_sdf(mask_a)
        sdf_b = self._compute_signed_sdf(mask_b)

        # Linear blend of distance fields
        return (1.0 - alpha) * sdf_a + alpha * sdf_b

    def get_grid_metadata(self) -> Dict:
        """Returns bounds & scale for downstream contour mapping."""
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "pixel_scale": self.pixel_scale,
            "grid_size": self.grid_size
        }
