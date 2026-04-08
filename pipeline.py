from scipy.interpolate import splev, splprep
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import triangle as tr
from scipy.sparse import csr_matrix
import json
from scipy.sparse.linalg import spsolve


class SFourierARAPMorpher:
    def __init__(self, pts_A, pts_B, quality=30):
        self.A = np.array(pts_A, dtype=np.float64)
        self.B = np.array(pts_B, dtype=np.float64)
        self.N = len(self.A)
        assert len(
            self.B) == self.N, "Both shapes must have the same number of points."

        self._build_mesh(quality)
        self._precompute_rest_geometry()
        self._precompute_fourier_path()

    def _build_mesh(self, quality):
        segments = np.stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)], axis=1)
        poly = {'vertices': self.A, 'segments': segments}
        mesh = tr.triangulate(poly, f'p -q{quality}')

        self.verts = mesh['vertices']
        self.triangles = mesh['triangles']
        self.n_verts = len(self.verts)

        self.boundary_idx = np.arange(self.N, dtype=np.int32)
        self.interior_idx = np.arange(self.N, self.n_verts, dtype=np.int32)

    @staticmethod
    def _cross2d(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def _precompute_rest_geometry(self):
        X0 = self.verts
        self.rest_verts = X0.copy()
        self.adj = [[] for _ in range(self.n_verts)]
        self.cot_weights = {}
        self.rest_edges = {}

        for tri in self.triangles:
            i, j, k = tri
            vi, vj, vk = X0[i], X0[j], X0[k]
            e_ij, e_jk, e_ki = vj - vi, vk - vj, vi - vk

            cot_i = np.dot(e_ki, e_ij) / \
                (np.abs(self._cross2d(e_ki, e_ij)) + 1e-10)
            cot_j = np.dot(e_ij, e_jk) / \
                (np.abs(self._cross2d(e_ij, e_jk)) + 1e-10)
            cot_k = np.dot(e_jk, e_ki) / \
                (np.abs(self._cross2d(e_jk, e_ki)) + 1e-10)

            for (u, v, cot) in [(i, j, cot_k), (j, i, cot_k),
                                (j, k, cot_i), (k, j, cot_i),
                                (k, i, cot_j), (i, k, cot_j)]:
                self.cot_weights[(u, v)] = self.cot_weights.get(
                    (u, v), 0.0) + cot
                self.adj[u].append(v)

        for u in range(self.n_verts):
            for v in self.adj[u]:
                self.rest_edges[(u, v)] = X0[v] - X0[u]

        rows, cols, vals = [], [], []
        for i in range(self.n_verts):
            diag = 0.0
            for j in self.adj[i]:
                w = self.cot_weights.get((i, j), 0.0)
                diag += w
                rows.extend([i, i])
                cols.extend([i, j])
                vals.extend([w, -w])
            rows.append(i)
            cols.append(i)
            vals.append(-diag)
        self.L = csr_matrix((vals, (rows, cols)),
                            shape=(self.n_verts, self.n_verts))

        self.L_ii = self.L[np.ix_(self.interior_idx, self.interior_idx)]
        self.L_ib = self.L[np.ix_(self.interior_idx, self.boundary_idx)]

    def _precompute_fourier_path(self):
        # Center & align phase/scale once for stable linear interpolation
        cA, cB = self.A.mean(0), self.B.mean(0)
        zA = (self.A - cA) @ [1.0, 1.0j]
        zB = (self.B - cB) @ [1.0, 1.0j]

        # Phase alignment (minimize L2 distance)
        best_k, min_err = 0, np.inf
        for k in range(self.N):
            err = np.sum(np.abs(zA - np.roll(zB, k))**2)
            if err < min_err:
                min_err, best_k = err, k
        zB = np.roll(zB, best_k)

        # Scale alignment
        scale = np.linalg.norm(zA) / (np.linalg.norm(zB) + 1e-10)
        zB *= scale

        self.cA, self.cB = cA, cB
        self.FA = np.fft.fft(zA)
        self.FB = np.fft.fft(zB)

    def _boundary_at_t(self, t):
        """Compute boundary contour at absolute progress t ∈ [0, 1]"""
        F_t = (1.0 - t) * self.FA + t * self.FB
        z_t = np.fft.ifft(F_t)
        cent = (1.0 - t) * self.cA + t * self.cB
        return np.c_[z_t.real, z_t.imag] + cent

    def _solve_frame(self, X, target_t, max_inner_iters=5, tol=1e-4):
        """
        Solve ARAP to match the boundary at `target_t`.
        Returns (deformed_mesh, target_boundary)
        """
        X_next = X.copy()
        target_b = self._boundary_at_t(target_t)

        for _ in range(max_inner_iters):
            # 1. Hard boundary constraint
            X_next[self.boundary_idx] = target_b

            # 2. Local Step: Optimal rotations
            R_list = []
            for i in self.interior_idx:
                nbrs = np.array(self.adj[i])
                if len(nbrs) == 0:
                    R_list.append(np.eye(2))
                    continue

                d0 = np.array([self.rest_edges[(i, j)] for j in nbrs])
                d = X_next[nbrs] - X_next[i]
                w = np.array([self.cot_weights.get((i, j), 0.0) for j in nbrs])

                C = d.T @ (w[:, None] * d0)
                U, _, Vt = np.linalg.svd(C)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, 1] *= -1
                    R = U @ Vt
                R_list.append(R)
            R_list = np.array(R_list)

            # 3. Global Step: Solve sparse system
            b_int = np.zeros((len(self.interior_idx), 2))
            for idx, i in enumerate(self.interior_idx):
                nbrs = np.array(self.adj[i])
                d0 = np.array([self.rest_edges[(i, j)] for j in nbrs])
                w = np.array([self.cot_weights.get((i, j), 0.0) for j in nbrs])
                b_int[idx] = (w[:, None] * (R_list[idx] @ d0.T).T).sum(axis=0)

            rhs = b_int - self.L_ib @ target_b
            X_int_new = spsolve(self.L_ii, rhs).reshape(-1, 2)

            delta = np.max(np.linalg.norm(
                X_int_new - X_next[self.interior_idx], axis=1))
            X_next[self.interior_idx] = X_int_new
            if delta < tol:
                break

        return X_next, target_b

    def _check_validity(self, X, target_b):
        tri_verts = X[self.triangles]
        a, b, c = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        areas = 0.5 * np.abs(self._cross2d(b - a, c - a))
        if np.any(areas < 1e-6):
            return False, "Degenerate triangles"

        for i in range(self.N):
            p1, p2 = target_b[i], target_b[(i+1) % self.N]
            for j in range(i+2, min(i+self.N-2, self.N)):
                p3, p4 = target_b[j], target_b[(j+1) % self.N]
                if self._segments_intersect(p1, p2, p3, p4):
                    return False, "Boundary self-intersection"
        return True, "Valid"

    @staticmethod
    def _ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

    def _segments_intersect(self, p1, p2, p3, p4):
        return (self._ccw(p1, p3, p4) != self._ccw(p2, p3, p4)) and \
               (self._ccw(p1, p2, p3) != self._ccw(p1, p2, p4))


class FourierARAPMorpher:
    def __init__(self, pts_A, pts_B, quality=30):
        self.A = np.array(pts_A, dtype=np.float64)
        self.B = np.array(pts_B, dtype=np.float64)
        self.N = len(self.A)
        assert len(
            self.B) == self.N, "Both shapes must have the same number of points."

        self._build_mesh(quality)
        self._precompute_rest_geometry()

    def _build_mesh(self, quality):
        segments = np.stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)], axis=1)
        poly = {'vertices': self.A, 'segments': segments}
        mesh = tr.triangulate(poly, f'p -q{quality}')

        self.verts = mesh['vertices']
        self.triangles = mesh['triangles']
        self.n_verts = len(self.verts)

        self.boundary_idx = np.arange(self.N, dtype=np.int32)
        self.interior_idx = np.arange(self.N, self.n_verts, dtype=np.int32)
        self.boundary_mask = np.zeros(self.n_verts, dtype=bool)
        self.boundary_mask[:self.N] = True

    @staticmethod
    def _cross2d(a, b):
        """NumPy 2.0 safe 2D cross product: a_x*b_y - a_y*b_x"""
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def _precompute_rest_geometry(self):
        X0 = self.verts
        self.rest_verts = X0.copy()
        self.rest_areas = np.zeros(len(self.triangles))
        self.rest_edges = {}

        self.adj = [[] for _ in range(self.n_verts)]
        self.cot_weights = {}

        for t_idx, tri in enumerate(self.triangles):
            i, j, k = tri
            vi, vj, vk = X0[i], X0[j], X0[k]

            # Rest area (2D cross product)
            self.rest_areas[t_idx] = 0.5 * \
                np.abs(self._cross2d(vj - vi, vk - vi))

            # Cotangent weights
            edges = [(i, j, vk), (j, k, vi), (k, i, vj)]
            for u, v, w in edges:
                e1, e2 = w - u, w - v
                cross_val = np.linalg.norm(self._cross2d(e1, e2))
                dot_val = np.dot(e1, e2)
                cot = dot_val / (cross_val + 1e-10)
                self.cot_weights[(u, v)] = self.cot_weights.get(
                    (u, v), 0.0) + cot
                self.cot_weights[(v, u)] = self.cot_weights.get(
                    (v, u), 0.0) + cot
                self.rest_edges[(u, v)] = X0[v] - X0[u]
                self.adj[u].append(v)

        self.interior_neighbors = {i: np.array(
            self.adj[i]) for i in self.interior_idx}

    def _adaptive_fourier_step(self, curr_boundary, dt):
        c_cent = np.mean(curr_boundary, axis=0)
        t_cent = np.mean(self.B, axis=0)

        z_c = (curr_boundary - c_cent) @ [1.0, 1.0j]
        z_t = (self.B - t_cent) @ [1.0, 1.0j]

        best_k, min_err = 0, np.inf
        for k in range(self.N):
            err = np.sum(np.abs(z_c - np.roll(z_t, k))**2)
            if err < min_err:
                min_err, best_k = err, k
        z_t = np.roll(z_t, best_k)

        scale = np.linalg.norm(z_c) / (np.linalg.norm(z_t) + 1e-10)
        z_t *= scale

        F_c = np.fft.fft(z_c)
        F_t = np.fft.fft(z_t)
        F_interp = (1.0 - dt) * F_c + dt * F_t
        z_interp = np.fft.ifft(F_interp)

        cent_interp = (1.0 - dt) * c_cent + dt * t_cent
        return np.c_[z_interp.real, z_interp.imag] + cent_interp

    def _project_rigidity(self, X, X_new):
        for i in self.interior_idx:
            nbrs = self.interior_neighbors[i]
            if len(nbrs) == 0:
                continue

            d0 = self.rest_verts[nbrs] - self.rest_verts[i]
            d = X[nbrs] - X[i]

            w = np.array([self.cot_weights.get((i, j), 0.0) for j in nbrs])
            W = np.diag(w)
            C = d.T @ W @ d0

            U, S, Vt = np.linalg.svd(C)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, 1] *= -1
                R = U @ Vt

            d_target = (R @ d0.T).T
            X_new[i] = np.sum(w[:, None] * (X[nbrs] - d_target),
                              axis=0) / (np.sum(w) + 1e-10)

    def _solve_frame(self, X, dt, max_inner_iters=10, tol=1e-3, damping=0.75):
        X_next = X.copy()

        for _ in range(max_inner_iters):
            target_b = self._adaptive_fourier_step(X[self.boundary_idx], dt)
            X_next[self.boundary_idx] = target_b

            X_rigid = X_next.copy()
            self._project_rigidity(X_next, X_rigid)

            X_area = np.zeros_like(X_next)
            tri_count = np.zeros(self.n_verts)

            tri_verts = X_rigid[self.triangles]
            centroids = np.mean(tri_verts, axis=1, keepdims=True)
            rel_vecs = tri_verts - centroids

            a, b, c = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
            curr_areas = 0.5 * np.abs(self._cross2d(b - a, c - a))

            scale = np.sqrt(self.rest_areas / (curr_areas + 1e-10))
            scale = np.clip(scale, 0.5, 2.0)[:, None, None]
            proj_tri = centroids + scale * rel_vecs

            for t_idx, (i, j, k) in enumerate(self.triangles):
                X_area[i] += proj_tri[t_idx, 0]
                X_area[j] += proj_tri[t_idx, 1]
                X_area[k] += proj_tri[t_idx, 2]
                tri_count[i] += 1
                tri_count[j] += 1
                tri_count[k] += 1

            mask = tri_count > 0
            X_area[mask] /= tri_count[mask, None]
            X_area[self.boundary_idx] = target_b

            delta = np.max(np.linalg.norm(X_area - X_next, axis=1))
            X_next = X_next + damping * (X_area - X_next)

            if delta < tol:
                break

        return X_next, target_b

    def _check_validity(self, X, target_b):
        min_area = np.min(self.rest_areas) * 0.05
        tri_verts = X[self.triangles]
        a, b, c = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        areas = 0.5 * np.abs(self._cross2d(b - a, c - a))
        if np.any(areas < min_area):
            return False, f"Degenerate triangles (min: {np.min(areas):.2e})"

        for i in range(self.N):
            p1, p2 = target_b[i], target_b[(i+1) % self.N]
            for j in range(i+2, min(i+self.N-2, self.N)):
                p3, p4 = target_b[j], target_b[(j+1) % self.N]
                if self._segments_intersect(p1, p2, p3, p4):
                    return False, "Boundary self-intersection"
        return True, "Valid"

    @staticmethod
    def _ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

    def _segments_intersect(self, p1, p2, p3, p4):
        return (self._ccw(p1, p3, p4) != self._ccw(p2, p3, p4)) and \
               (self._ccw(p1, p2, p3) != self._ccw(p1, p2, p4))


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


def get_bezier(p0, p1, p2, steps=12):
    t = np.linspace(0, 1, steps)[:, None]
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2


def bezier_line(pts, steps=12):
    out = []
    n = len(pts)
    for i in range(n):
        out.append(get_bezier(pts[i], pts[(i+1) % n], pts[(i+2) % n], steps))
    return np.vstack(out)


def s_run_morph_test(A, B, output_path="morph_arap.gif"):
    print(f"🔹 Initialized ARAP morpher on {len(A)} control points...")
    morpher = SFourierARAPMorpher(A, B, quality=30)

    print("🔹 Simulating game loop (61 frames)...")
    frames = []
    X_mesh = morpher.verts.copy()  # 👈 Start with FULL mesh (boundary + interior)

    for i in range(61):
        t = i / 60.0
        t_ease = t**2 * (3 - 2*t)  # Smoothstep

        # Pass absolute progress `t_ease` instead of relative `dt`
        X_mesh, target_b = morpher._solve_frame(
            X_mesh, target_t=t_ease, max_inner_iters=5, tol=1e-4
        )
        frames.append(target_b.copy())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    curve, = ax.plot([], [], 'b-', lw=2.2)
    anc,   = ax.plot([], [], 'ro', ms=3)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        pts = frames[f]
        poly = bezier_line(pts)
        curve.set_data(poly[:, 0], poly[:, 1])
        anc.set_data(pts[:, 0], pts[:, 1])
        title.set_text(f"t = {f/60:.2f} | {len(pts)} pts")
        return curve, anc, title

    print("🔹 Rendering GIF...")
    anim = FuncAnimation(fig, update, frames=61, blit=False)
    anim.save(output_path, fps=30, writer='pillow')
    print(f"✅ Saved: {output_path}")
    plt.close(fig)


def run_morph_test(A, B, output_path="morph_arap.gif"):
    print(f"🔹 Initialized ARAP morpher on {len(A)} control points...")
    morpher = FourierARAPMorpher(A, B, quality=30)

    print("🔹 Simulating game loop (61 frames)...")
    frames = []
    X_mesh = morpher.verts.copy()  # 👈 Start with FULL mesh (boundary + interior)

    for i in range(61):
        # Solve returns updated full mesh + deformed boundary
        X_mesh, target_b = morpher._solve_frame(
            X_mesh, dt=1/60.0, max_inner_iters=8, tol=1e-3, damping=0.75
        )
        # 👈 Store ONLY the boundary for rendering
        frames.append(target_b.copy())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    curve, = ax.plot([], [], 'b-', lw=2.2)
    anc,   = ax.plot([], [], 'ro', ms=3)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=12)

    def update(f):
        pts = frames[f]
        poly = bezier_line(pts)
        curve.set_data(poly[:, 0], poly[:, 1])
        anc.set_data(pts[:, 0], pts[:, 1])
        title.set_text(f"t = {f/60:.2f} | {len(pts)} pts")
        return curve, anc, title

    print("🔹 Rendering GIF...")
    anim = FuncAnimation(fig, update, frames=61, blit=False)
    anim.save(output_path, fps=30, writer='pillow')
    print(f"✅ Saved: {output_path}")
    plt.close(fig)


TARGET_SEGMENTS = 64
# ---------------- Example Usage ----------------
if __name__ == "__main__":
    A = load_and_upgrade_v1("/workspace/C_shape.json", TARGET_SEGMENTS)
    B = load_and_upgrade_v1("/workspace/circle.json", TARGET_SEGMENTS)

    print(
        f"🔹 Initialized morpher on {len(A)} control points ({TARGET_SEGMENTS} quad curves)")

    run_morph_test(A, B, "morph_test.gif")
