# mesh_builder.py — ANIMATION-READY FINAL VERSION
import numpy as np
import triangle as tr
from typing import List, Tuple


class MeshBuilder:
    def __init__(self, pts_A: np.ndarray, min_angle: float = 28.0, shell_thickness: float = 0.06):
        """
        pts_A: CCW-ordered boundary points (your main shape)
        shell_thickness: thickness of the band (0.06+ for visibility)
        """
        self.A = np.asarray(pts_A, dtype=float)
        self.N = len(self.A)
        self.shell_thickness = shell_thickness

        # 👇 Step 1: Create single continuous "racetrack" boundary
        self._create_racetrack_boundary()

        # 👇 Step 2: Triangulate ONLY the racetrack area
        self._build_mesh(min_angle)

        # 👇 Step 3: Extract edges
        self._extract_edges()

        # --------------------------
        # CRITICAL FIX: MAP BACK TO ORIGINAL CCW ORDER
        # The inner loop is reversed CW, so we reverse it back to CCW
        # --------------------------
        self.boundary_idx = np.arange(2*self.N - 1, self.N - 1, -1)

    def _create_racetrack_boundary(self):
        """
        Create a SINGLE CONTINUOUS BOUNDARY:
        1. Outer loop: CCW, offset outward
        2. Inner loop: CW, original shape reversed
        """
        pts = self.A

        # Compute OUTWARD normals for CCW shape
        vel = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        # 90° CW = outward for CCW
        normal = np.stack([vel[:, 1], -vel[:, 0]], axis=1)
        norm_len = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        normal = normal / norm_len

        # 1. Outer loop (CCW): offset outward
        outer_pts = pts + normal * self.shell_thickness

        # 2. Inner loop (CW): original shape, reversed
        inner_pts = pts[::-1].copy()

        # Full vertices: [OUTER (CCW), INNER (CW)]
        self.full_vertices = np.vstack([outer_pts, inner_pts])
        self.N_total = len(self.full_vertices)

        # Build a SINGLE CONTINUOUS SEGMENT LOOP
        segs_outer = np.column_stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)])
        segs_inner = np.column_stack(
            [np.arange(self.N, 2*self.N), np.roll(np.arange(self.N, 2*self.N), -1)])
        self.segments = np.vstack([segs_outer, segs_inner])

    def _build_mesh(self, min_angle):
        data = {
            'vertices': self.full_vertices,
            'segments': self.segments,
            'segment_markers': np.ones(len(self.segments), dtype=int)
        }
        mesh = tr.triangulate(data, f'pa{min_angle:.1f}qez')
        self.verts = mesh['vertices'].astype(np.float32)
        self.triangles = mesh['triangles']
        self.n_verts = len(self.verts)

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
