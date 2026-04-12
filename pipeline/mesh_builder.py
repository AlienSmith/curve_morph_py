# mesh_builder.py — DEBUG VERSION
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from typing import List, Tuple


class MeshBuilder:
    def __init__(self, pts_A: np.ndarray, min_angle: float = 28.0, shell_thickness: float = 0.08):
        """
        pts_A: CCW-ordered boundary points (your main shape)
        shell_thickness: MAKE THIS LARGE (0.08+) for visibility
        """
        self.A = np.asarray(pts_A, dtype=float)
        self.N = len(self.A)
        self.shell_thickness = shell_thickness

        print(f"[DEBUG] Input points: {self.N}")
        print(f"[DEBUG] Shell thickness: {self.shell_thickness}")

        # 👇 Step 1: Create CCW outer + CW inner boundaries
        self._create_donut_boundaries()

        # 👇 Step 2: Define a VERY OBVIOUS hole point
        self._define_hole()

        # 👇 Step 3: Plot raw input BEFORE triangulation (DEBUG)
        self._plot_raw_input()

        # 👇 Step 4: Triangulate
        self._build_mesh(min_angle)

        # 👇 Step 5: Extract edges
        self._extract_edges()

        # Map original input points (now reversed CW inner loop) back to CCW order
        self.boundary_idx = np.arange(2*self.N - 1, self.N - 1, -1)
        print(f"[DEBUG] Boundary indices: {self.boundary_idx[:5]}...")

    def _create_donut_boundaries(self):
        """
        STANDARD DONUT TOPOLOGY FOR TRIANGLE:
        1. OUTER LOOP: CCW, offset outward
        2. INNER LOOP: CW, original shape reversed
        """
        pts = self.A

        # Compute OUTWARD normals for CCW shape
        vel = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        # 90° CW = outward for CCW
        normal = np.stack([vel[:, 1], -vel[:, 0]], axis=1)
        norm_len = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        normal = normal / norm_len

        # 1. OUTER LOOP (CCW): offset outward by a LARGE amount
        outer_pts = pts + normal * self.shell_thickness

        # 2. INNER LOOP (CW): original shape, REVERSED
        inner_pts = pts[::-1].copy()

        # Full vertices: [OUTER (CCW), INNER (CW)]
        self.full_vertices = np.vstack([outer_pts, inner_pts])
        self.N_total = len(self.full_vertices)

        # Build CLOSED segments for both loops
        segs_outer = np.column_stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)])
        segs_inner = np.column_stack(
            [np.arange(self.N, 2*self.N), np.roll(np.arange(self.N, 2*self.N), -1)])
        self.segments = np.vstack([segs_outer, segs_inner])

        print(f"[DEBUG] Total vertices: {self.N_total}")
        print(f"[DEBUG] Total segments: {len(self.segments)}")

    def _define_hole(self):
        """
        HOLE POINT: Pick a point that is DEFINITELY inside the original shape.
        For a C-shape, centroid might be outside — so we pick the FIRST point's neighbor.
        """
        # Pick a point 1/3 of the way between point 0 and point 1 (definitely inside)
        hole_candidate = 0.6 * self.A[0] + 0.4 * self.A[1]
        self.hole = hole_candidate[None, :]
        print(f"[DEBUG] Hole point: {self.hole}")

    def _plot_raw_input(self):
        """Plot raw vertices, segments, and hole BEFORE triangulation (DEBUG)."""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        # Plot all vertices
        ax.scatter(self.full_vertices[:, 0], self.full_vertices[:,
                   1], s=10, c='gray', label='All Vertices')

        # Plot outer loop (CCW) in red
        outer_verts = self.full_vertices[:self.N]
        ax.plot(outer_verts[:, 0], outer_verts[:, 1],
                c='red', lw=2, label='Outer Loop (CCW)')
        ax.scatter(outer_verts[0, 0], outer_verts[0, 1],
                   s=50, c='red', marker='X', label='Outer Start')

        # Plot inner loop (CW) in blue
        inner_verts = self.full_vertices[self.N:]
        ax.plot(inner_verts[:, 0], inner_verts[:, 1],
                c='blue', lw=2, label='Inner Loop (CW)')
        ax.scatter(inner_verts[0, 0], inner_verts[0, 1],
                   s=50, c='blue', marker='X', label='Inner Start')

        # Plot hole point in green
        ax.scatter(self.hole[0, 0], self.hole[0, 1], s=100,
                   c='limegreen', marker='o', label='Hole Point')

        ax.set_aspect('equal')
        ax.legend()
        ax.set_title("Raw Input (Check This First!)")
        plt.savefig("debug_raw_input.png")
        print("✅ Saved debug plot: debug_raw_input.png")
        plt.close()

    def _build_mesh(self, min_angle):
        """Triangulate with simple, safe flags."""
        data = {
            'vertices': self.full_vertices,
            'segments': self.segments,
            'segment_markers': np.ones(len(self.segments), dtype=int),
            'holes': self.hole
        }

        print("[DEBUG] Starting triangulation...")
        mesh = tr.triangulate(data, f'pq{min_angle:.1f}ez')
        print("[DEBUG] Triangulation complete!")

        self.verts = mesh['vertices'].astype(np.float32)
        self.triangles = mesh['triangles']
        self.n_verts = len(self.verts)
        print(f"[DEBUG] Triangulated vertices: {self.n_verts}")
        print(f"[DEBUG] Triangles: {len(self.triangles)}")

    def _extract_edges(self):
        edges = set()
        for tri in self.triangles:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i+1) % 3])
                edges.add((a, b) if a < b else (b, a))
        self.edges = list(edges)
        self.rest_lengths = np.array([
            np.linalg.norm(self.verts[i] - self.verts[j])
            for i, j in self.edges
        ], dtype=np.float32)

    def get_interior_indices(self) -> np.ndarray:
        all_idx = np.arange(self.n_verts)
        return np.setdiff1d(all_idx, self.boundary_idx)
