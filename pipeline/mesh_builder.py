# mesh_builder.py — FULL DOUBLE-LAYER VERSION
import numpy as np
import triangle as tr
from typing import List, Tuple


class MeshBuilder:
    def __init__(self, pts_A: np.ndarray, min_angle: float = 28.0, shell_thickness: float = 0.025):
        self.A = np.asarray(pts_A, dtype=float)
        self.N = len(self.A)
        self.shell_thickness = shell_thickness  # Thin solid layer

        # 👇 CREATE DOUBLE LAYER (INNER + OUTER BOUNDARY)
        self._create_shell_boundary()
        self._build_mesh(min_angle)
        self._extract_edges()

        # ORIGINAL INPUT POINTS = MAIN BOUNDARY (we drive this with Fourier)
        self.boundary_idx = np.arange(self.N)

    def _create_shell_boundary(self):
        """Create inner + outer offset to make a THIN SOLID SHELL."""
        pts = self.A

        # Compute OUTWARD NORMALS for offset
        vel = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        normal = np.stack([vel[:, 1], -vel[:, 0]], axis=1)
        norm_len = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        normal = normal / norm_len

        # Outer boundary (offset outward)
        outer_pts = pts + normal * self.shell_thickness

        # Full vertices: [ORIGINAL (inner), OUTER]
        self.full_vertices = np.vstack([pts, outer_pts])
        self.N_shell = len(self.full_vertices)

        # Build closed segments for triangulation
        segs_inner = np.column_stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)])
        segs_outer = np.column_stack(
            [np.arange(self.N)+self.N, np.roll(np.arange(self.N)+self.N, -1)])
        segs_bridge = np.column_stack(
            [np.arange(self.N), np.arange(self.N)+self.N])

        self.segments = np.vstack([segs_inner, segs_outer, segs_bridge])

    def _build_mesh(self, min_angle):
        data = {
            'vertices': self.full_vertices,
            'segments': self.segments,
            'segment_markers': np.ones(len(self.segments), dtype=int)
        }
        # Generate quality triangulation of the SHELL
        mesh = tr.triangulate(data, f'pq{min_angle:.1f}ez')

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
