import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
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


# ==============================
# FFT MORPHER (Now operates on higher-res)
# ==============================


# class FourierShapeMorpher:
#     def __init__(self, pts_A, pts_B, n_segments=1, spectral_smooth=0.15):
#         self.A = np.asarray(pts_A, dtype=np.float32)
#         self.B = np.asarray(pts_B, dtype=np.float32)
#         self.n_points = len(self.A)
#         self.n_segments = max(1, int(n_segments))
#         self.spectral_smooth = spectral_smooth
#         self._precompute()

#     def _precompute(self):
#         N = self.n_points
#         Az = self.A[:, 0] + 1j * self.A[:, 1]
#         Bz = self.B[:, 0] + 1j * self.B[:, 1]

#         self.centroid_A = Az.mean()
#         self.centroid_B = Bz.mean()

#         A_c = Az - self.centroid_A
#         B_c = Bz - self.centroid_B

#         # 1. Energy normalization (keeps scale consistent)
#         self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
#         B_c *= self.scale

#         # 2. Deterministic cyclic alignment (breaks circle symmetry)
#         corr = np.abs(np.fft.ifft(np.fft.fft(A_c) * np.conj(np.fft.fft(B_c))))
#         # Gaussian bias strongly penalizes large phase shifts, preventing arbitrary 180° flips
#         bias = np.exp(-0.5 * (np.arange(N) / (N * 0.2))**2)
#         self.phase_shift = np.argmax(corr * bias)
#         B_c = np.roll(B_c, -self.phase_shift)

#         # 3. Fourier descriptors + spectral taper
#         self.window = self._smooth_spectral_window(N)
#         self.FA = np.fft.fft(A_c) * self.window
#         self.FB = np.fft.fft(B_c) * self.window

#         # 4. Precompute segment boundaries using stable Cartesian interpolation
#         self.seg_FAs, self.seg_FBs = [], []
#         self.seg_cA, self.seg_cB = [], []
#         boundary_ts = np.linspace(0.0, 1.0, self.n_segments + 1)

#         for i in range(self.n_segments):
#             t0, t1 = boundary_ts[i], boundary_ts[i+1]
#             self.seg_cA.append((1-t0)*self.centroid_A + t0*self.centroid_B)
#             self.seg_cB.append((1-t1)*self.centroid_A + t1*self.centroid_B)
#             self.seg_FAs.append(
#                 self._interpolate_complex(self.FA, self.FB, t0))
#             self.seg_FBs.append(
#                 self._interpolate_complex(self.FA, self.FB, t1))

#     def _smooth_spectral_window(self, N):
#         """Gaussian taper to suppress unstable high frequencies."""
#         k = np.arange(N)
#         k = np.minimum(k, N - k)
#         sigma = N * self.spectral_smooth
#         return np.exp(-0.5 * (k / sigma)**2)

#     def _interpolate_complex(self, F1, F2, t):
#         """
#         Robust Cartesian interpolation with zero-crossing stabilization.
#         Replaces polar interpolation to prevent independent harmonic phase twisting.
#         """
#         # Linear blend in complex plane preserves relative phase between all harmonics
#         Ft = (1.0 - t) * F1 + t * F2

#         # Prevent magnitude collapse near zero without disrupting phase coherence
#         mags = np.abs(Ft)
#         floor = 1e-6
#         mask = mags < floor
#         if np.any(mask):
#             Ft[mask] = floor * np.exp(1j * np.angle(F1[mask]))

#         return Ft

#     def evaluate(self, t_ease):
#         t_ease = np.clip(t_ease, 0.0, 1.0)

#         if self.n_segments == 1:
#             t_blend = 3 * t_ease**2 - 2 * t_ease**3  # Smoothstep for spatial blending
#             Ft = self._interpolate_complex(self.FA, self.FB, t_ease)
#             z = np.fft.ifft(Ft) + (1 - t_blend) * self.centroid_A + \
#                 t_blend * self.centroid_B
#             return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]

#         # Segment mapping
#         seg_idx = int(t_ease * self.n_segments)
#         seg_idx = min(seg_idx, self.n_segments - 1)
#         local_t = (t_ease * self.n_segments) - seg_idx
#         t_spatial = 3 * local_t**2 - 2 * local_t**3

#         FA, FB = self.seg_FAs[seg_idx], self.seg_FBs[seg_idx]
#         Ft = self._interpolate_complex(FA, FB, local_t)
#         z = np.fft.ifft(Ft) + (1 - t_spatial) * \
#             self.seg_cA[seg_idx] + t_spatial * self.seg_cB[seg_idx]

#         return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]


# class FourierShapeMorpher:
#     def __init__(self, pts_A, pts_B, n_segments=1, spectral_smooth=0.15):
#         self.A = np.asarray(pts_A, dtype=np.float32)
#         self.B = np.asarray(pts_B, dtype=np.float32)
#         self.n_points = len(self.A)
#         self.n_segments = max(1, int(n_segments))
#         self.spectral_smooth = spectral_smooth
#         self._precompute()

#     def _precompute(self):
#         N = self.n_points
#         Az = self.A[:, 0] + 1j * self.A[:, 1]
#         Bz = self.B[:, 0] + 1j * self.B[:, 1]

#         self.centroid_A = Az.mean()
#         self.centroid_B = Bz.mean()

#         A_c = Az - self.centroid_A
#         B_c = Bz - self.centroid_B

#         # 1. Energy matching (keeps your original shrink/expand behavior)
#         self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
#         B_c *= self.scale

#         # 2. Deterministic cyclic alignment (fixes left/right overlap shifting)
#         corr = np.abs(np.fft.ifft(np.fft.fft(A_c) * np.conj(np.fft.fft(B_c))))
#         # Strong bias toward 0-shift prevents arbitrary flips on symmetric shapes
#         bias = np.exp(-0.5 * (np.arange(N) / (N * 0.15))**2)
#         self.phase_shift = np.argmax(corr * bias)
#         B_c = np.roll(B_c, -self.phase_shift)

#         # 3. Fourier descriptors + spectral taper
#         self.window = self._smooth_spectral_window(N)
#         self.FA = np.fft.fft(A_c) * self.window
#         self.FB = np.fft.fft(B_c) * self.window

#         # 4. Precompute segment boundaries
#         self.seg_FAs, self.seg_FBs = [], []
#         self.seg_cA, self.seg_cB = [], []
#         boundary_ts = np.linspace(0.0, 1.0, self.n_segments + 1)

#         for i in range(self.n_segments):
#             t0, t1 = boundary_ts[i], boundary_ts[i+1]
#             self.seg_cA.append((1-t0)*self.centroid_A + t0*self.centroid_B)
#             self.seg_cB.append((1-t1)*self.centroid_A + t1*self.centroid_B)
#             self.seg_FAs.append(
#                 self._interpolate_complex(self.FA, self.FB, t0))
#             self.seg_FBs.append(
#                 self._interpolate_complex(self.FA, self.FB, t1))

#     def _smooth_spectral_window(self, N):
#         k = np.arange(N)
#         k = np.minimum(k, N - k)
#         sigma = N * self.spectral_smooth
#         return np.exp(-0.5 * (k / sigma)**2)

#     def _interpolate_complex(self, F1, F2, t):
#         """
#         Preserves original shrink/expand (geometric mean magnitude)
#         but uses coherent phase interpolation to prevent self-intersection.
#         """
#         # 1. Linear blend in complex plane maintains harmonic phase relationships
#         Ft_linear = (1.0 - t) * F1 + t * F2

#         # 2. Compute target magnitude to preserve your original shrink/expand curve
#         mag_target = np.maximum(np.abs(F1)**(1-t) * np.abs(F2)**t, 1e-6)
#         mag_linear = np.abs(Ft_linear)

#         # 3. Scale linear blend to match target magnitude, preserving phase
#         scale = np.where(mag_linear > 1e-7, mag_target / mag_linear, 1.0)
#         return Ft_linear * scale

#     def evaluate(self, t_ease):
#         t_ease = np.clip(t_ease, 0.0, 1.0)

#         if self.n_segments == 1:
#             t = 3 * t_ease**2 - 2 * t_ease**3  # Smoothstep for spatial blending
#             Ft = self._interpolate_complex(self.FA, self.FB, t_ease)
#             z = np.fft.ifft(Ft) + (1 - t) * self.centroid_A + \
#                 t * self.centroid_B
#             return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]

#         # Segment mapping
#         seg_idx = int(t_ease * self.n_segments)
#         seg_idx = min(seg_idx, self.n_segments - 1)
#         local_t = (t_ease * self.n_segments) - seg_idx
#         t_spatial = 3 * local_t**2 - 2 * local_t**3

#         FA, FB = self.seg_FAs[seg_idx], self.seg_FBs[seg_idx]
#         Ft = self._interpolate_complex(FA, FB, local_t)
#         z = np.fft.ifft(Ft) + (1 - t_spatial) * \
#             self.seg_cA[seg_idx] + t_spatial * self.seg_cB[seg_idx]

#         return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]


# class FourierShapeMorpher:
#     def __init__(self, pts_A, pts_B, n_segments=1, spectral_smooth=0.05):
#         self.A = np.asarray(pts_A, dtype=np.float32)
#         self.B = np.asarray(pts_B, dtype=np.float32)
#         self.n_points = len(self.A)
#         self.n_segments = max(1, int(n_segments))
#         self.spectral_smooth = spectral_smooth
#         self._precompute()

#     def _precompute(self):
#         N = self.n_points
#         Az = self.A[:, 0] + 1j * self.A[:, 1]
#         Bz = self.B[:, 0] + 1j * self.B[:, 1]

#         self.centroid_A = Az.mean()
#         self.centroid_B = Bz.mean()

#         A_c = Az - self.centroid_A
#         B_c = Bz - self.centroid_B

#         # 1. Energy matching
#         self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
#         B_c *= self.scale

#         # 2. Optimal cyclic alignment via cross-correlation
#         corr = np.abs(np.fft.ifft(np.fft.fft(A_c) * np.conj(np.fft.fft(B_c))))
#         # Gentle bias toward 0-shift breaks circle symmetry & prevents arbitrary twisting
#         corr *= np.exp(-0.5 * (np.arange(N) / (N * 0.25))**2)
#         self.phase_shift = np.argmax(corr)
#         B_c = np.roll(B_c, -self.phase_shift)

#         # 3. Fourier descriptors + spectral taper
#         self.window = self._smooth_spectral_window(N)
#         self.FA = np.fft.fft(A_c) * self.window
#         self.FB = np.fft.fft(B_c) * self.window

#         # 4. Precompute exact segment boundaries using continuous interpolation
#         self.seg_FAs, self.seg_FBs = [], []
#         self.seg_cA, self.seg_cB = [], []
#         boundary_ts = np.linspace(0.0, 1.0, self.n_segments + 1)

#         for i in range(self.n_segments):
#             t0, t1 = boundary_ts[i], boundary_ts[i+1]
#             self.seg_cA.append((1-t0)*self.centroid_A + t0*self.centroid_B)
#             self.seg_cB.append((1-t1)*self.centroid_A + t1*self.centroid_B)
#             self.seg_FAs.append(
#                 self._interpolate_complex(self.FA, self.FB, t0))
#             self.seg_FBs.append(
#                 self._interpolate_complex(self.FA, self.FB, t1))

#     def _smooth_spectral_window(self, N):
#         """Gaussian taper to suppress unstable high frequencies."""
#         k = np.arange(N)
#         k = np.minimum(k, N - k)
#         sigma = N * self.spectral_smooth
#         return np.exp(-0.5 * (k / sigma)**2)

#     def _interpolate_complex(self, F1, F2, t):
#         """
#         Cartesian interpolation preserves harmonic phase relationships.
#         Prevents independent frequency twisting that causes self-intersection.
#         """
#         # Linear blend in complex domain (much more stable than polar interpolation)
#         Ft = (1.0 - t) * F1 + t * F2

#         # Magnitude floor to prevent numerical instability near zero-crossings
#         mags = np.abs(Ft)
#         floor = 1e-6
#         mask = mags < floor
#         if np.any(mask):
#             # Fallback to source phase to avoid sudden 180° phase flips
#             Ft[mask] = floor * np.exp(1j * np.angle(F1[mask]))

#         return Ft

#     def evaluate(self, t_ease):
#         t_ease = np.clip(t_ease, 0.0, 1.0)

#         if self.n_segments == 1:
#             # Linear for Fourier (stable harmonic blending)
#             # Smoothstep for centroid (natural visual easing)
#             t_fourier = t_ease
#             t_spatial = 3 * t_ease**2 - 2 * t_ease**3

#             Ft = self._interpolate_complex(self.FA, self.FB, t_fourier)
#             z = np.fft.ifft(Ft) + (1 - t_spatial) * self.centroid_A + \
#                 t_spatial * self.centroid_B
#             return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]

#         # Segment mapping
#         seg_idx = int(t_ease * self.n_segments)
#         seg_idx = min(seg_idx, self.n_segments - 1)
#         local_t = (t_ease * self.n_segments) - seg_idx

#         t_fourier = local_t
#         t_spatial = 3 * local_t**2 - 2 * local_t**3

#         FA, FB = self.seg_FAs[seg_idx], self.seg_FBs[seg_idx]
#         Ft = self._interpolate_complex(FA, FB, t_fourier)
#         z = np.fft.ifft(Ft) + (1 - t_spatial) * \
#             self.seg_cA[seg_idx] + t_spatial * self.seg_cB[seg_idx]

#         return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]


# class FourierShapeMorpher:
#     def __init__(self, pts_A, pts_B, n_segments=1):
#         self.A = np.asarray(pts_A, dtype=np.float32)
#         self.B = np.asarray(pts_B, dtype=np.float32)
#         self.n_points = len(pts_A)
#         self.n_segments = max(1, int(n_segments))
#         self._precompute()
#         print(f"Segments initialized: {self.n_segments}")

#     def _slerp(self, F1, F2, t):
#         """球面线性插值傅里叶描述子，保证幅度与相位平滑过渡"""
#         norm1, norm2 = np.linalg.norm(F1), np.linalg.norm(F2)
#         if norm1 < 1e-8 or norm2 < 1e-8:
#             return (1 - t) * F1 + t * F2

#         F1_n, F2_n = F1 / norm1, F2 / norm2
#         cos_om = np.clip(np.real(np.sum(F1_n * np.conj(F2_n))), -1.0, 1.0)
#         omega = np.arccos(cos_om)

#         if omega < 1e-6:
#             return F1 * (1 - t) + F2 * t

#         sin_om = np.sin(omega)
#         c1 = np.sin((1 - t) * omega) / sin_om * norm1
#         c2 = np.sin(t * omega) / sin_om * norm2
#         return c1 * F1_n + c2 * F2_n

#     def _precompute(self):
#         N = self.n_points
#         Az = self.A[:, 0] + 1j * self.A[:, 1]
#         Bz = self.B[:, 0] + 1j * self.B[:, 1]

#         self.centroid_A = Az.mean()
#         self.centroid_B = Bz.mean()

#         A_c = Az - self.centroid_A
#         B_c = Bz - self.centroid_B

#         self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
#         B_c *= self.scale

#         # 最优起始相位对齐
#         best_err, best_k = np.inf, 0
#         for k in range(N):
#             err = np.sum(np.abs(A_c - np.roll(B_c, k))**2)
#             if err < best_err:
#                 best_err, best_k = err, k
#         B_c = np.roll(B_c, best_k)

#         self.FA = np.fft.fft(A_c)
#         self.FB = np.fft.fft(B_c)
#         self.phase_shift = best_k

#         # 🔑 核心修复：直接在频域计算分段边界，避免 ifft->fft 往返误差
#         self.seg_FAs = []
#         self.seg_FBs = []
#         self.seg_cA = []
#         self.seg_cB = []

#         boundary_ts = np.linspace(0.0, 1.0, self.n_segments + 1)
#         for i in range(self.n_segments):
#             t_start = boundary_ts[i]
#             t_end = boundary_ts[i + 1]

#             self.seg_cA.append(
#                 (1 - t_start) * self.centroid_A + t_start * self.centroid_B)
#             self.seg_cB.append(
#                 (1 - t_end) * self.centroid_A + t_end * self.centroid_B)
#             self.seg_FAs.append(self._slerp(self.FA, self.FB, t_start))
#             self.seg_FBs.append(self._slerp(self.FA, self.FB, t_end))

#     def evaluate(self, t_ease):
#         t_ease = np.clip(t_ease, 0.0, 1.0)

#         if self.n_segments == 1:
#             # 单段直接使用全局 Slerp + Smoothstep
#             t = 3 * t_ease**2 - 2 * t_ease**3
#             # 注意：slerp 本身用线性 t，easing 仅作用于逆变换后的坐标插值逻辑
#             Ft = self._slerp(self.FA, self.FB, t_ease)
#             # 但通常我们希望 easing 作用于整个路径。更标准的做法是 slerp 也用 easing 后的 t：
#             # Ft = self._slerp(self.FA, self.FB, t_ease) 保持原逻辑即可，下面会修正
#             pass

#         # 🔑 分段映射
#         seg_idx = int(t_ease * self.n_segments)
#         seg_idx = min(seg_idx, self.n_segments - 1)
#         local_t = (t_ease * self.n_segments) - seg_idx

#         # 段内 Smoothstep（保证边界处一阶导数为 0，实现视觉无缝）
#         t = 3 * local_t**2 - 2 * local_t**3

#         FA, FB = self.seg_FAs[seg_idx], self.seg_FBs[seg_idx]
#         # Slerp 使用线性 local_t 保持数学连续性，easing 已通过坐标混合体现
#         # 若希望整体速度曲线一致，也可将 slerp 的 t 替换为 t_ease 映射值，但通常用线性即可
#         Ft = self._slerp(FA, FB, local_t)

#         z = np.fft.ifft(Ft) + (1 - t) * \
#             self.seg_cA[seg_idx] + t * self.seg_cB[seg_idx]
#         return np.c_[z.real.astype(np.float32), z.imag.astype(np.float32)]


class FourierShapeMorpher:
    def __init__(self, pts_A, pts_B):
        self.A = pts_A
        self.B = pts_B
        self.n_points = len(pts_A)
        self._precompute()

    def _precompute(self):
        N = self.n_points
        Az = self.A[:, 0] + 1j * self.A[:, 1]
        Bz = self.B[:, 0] + 1j * self.B[:, 1]

        self.centroid_A = Az.mean()
        self.centroid_B = Bz.mean()

        A_c = Az - self.centroid_A
        B_c = Bz - self.centroid_B

        self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
        B_c *= self.scale

        best_err, best_k = np.inf, 0
        for k in range(N):
            err = np.sum(np.abs(A_c - np.roll(B_c, k))**2)
            if err < best_err:
                best_err, best_k = err, k
        B_c = np.roll(B_c, best_k)

        self.FA = np.fft.fft(A_c)
        self.FB = np.fft.fft(B_c)
        self.phase_shift = best_k

    def evaluate(self, t_ease):
        Ft = (1.0 - t_ease) * self.FA + t_ease * self.FB
        z_t = np.fft.ifft(Ft)
        centroid = (1.0 - t_ease) * self.centroid_A + t_ease * self.centroid_B
        z_t += centroid
        return np.c_[z_t.real.astype(np.float32), z_t.imag.astype(np.float32)]

# ==============================
# RENDER & TEST
# ==============================


def run_morph_test():
    print(f"🔹 Loading & upgrading shapes to {TARGET_SEGMENTS} segments...")
    # 👇 Replace with your actual filenames
    A = load_and_upgrade("/workspace/Shape_A.json", TARGET_SEGMENTS)
    B = load_and_upgrade("/workspace/Shape_B.json", TARGET_SEGMENTS)

    print(
        f"🔹 Initialized morpher on {len(A)} control points ({TARGET_SEGMENTS} quad curves)")
    morpher = FourierShapeMorpher(A, B)

    print("🔹 Simulating game loop (61 frames)...")
    anim = []
    for i in range(61):
        t = i / 60.0
        t_ease = t**2 * (3 - 2*t)
        pts = morpher.evaluate(t_ease)
        anim.append(pts)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    curve, = ax.plot([], [], 'b-', lw=2.2)
    anc,   = ax.plot([], [], 'ro', ms=3)  # Smaller dots for higher res
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        pts = anim[f]
        poly = bezier_line(pts)
        curve.set_data(poly[:, 0], poly[:, 1])
        anc.set_data(pts[:, 0], pts[:, 1])
        title.set_text(f"t = {f/60:.2f} | {len(pts)} pts")
        return curve, anc, title

    print("🔹 Rendering GIF...")
    FuncAnimation(fig, update, frames=61, blit=False).save(
        "morph_highres.gif", fps=30, writer='pillow')
    print("✅ Saved: morph_highres.gif")


if __name__ == "__main__":
    run_morph_test()
