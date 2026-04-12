import numpy as np
import triangle as tr
from typing import List, Tuple


class MeshBuilder:
    def __init__(self, pts_A: np.ndarray, min_angle: float = 30.0, offset: float = 0.02):
        self.A = np.asarray(pts_A, dtype=float)
        self.N = len(self.A)
        self.offset = offset  # Thin shell thickness

        # 👇 CREATE DOUBLE LAYER (ORIGINAL + OFFSET OUTLINE)
        self._create_double_layer()
        self._build_mesh(min_angle)
        self._extract_edges()

        # Original boundary is still the first N points
        self.boundary_idx = np.arange(self.N)

    def _create_double_layer(self):
        """Create inner + outer boundary (thin shell)."""
        pts = self.A
        # Compute normals pointing outward
        tgt = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        norm = np.stack([-tgt[:, 1], tgt[:, 0]], axis=1)
        norm = norm / (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-8)

        # Outer layer (offset outward)
        outer = pts + norm * self.offset
        # Combine: original (inner) + outer → closed loop
        self.verts_full = np.concatenate([pts, outer])
        self.N_full = len(self.verts_full)

        # Segments for double closed loop
        segs_inner = np.stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)], axis=1)
        segs_outer = np.stack(
            [np.arange(self.N)+self.N, np.roll(np.arange(self.N)+self.N, -1)], axis=1)
        segs_connect = np.stack(
            [np.arange(self.N), np.arange(self.N)+self.N], axis=1)
        self.segs = np.concatenate([segs_inner, segs_outer, segs_connect])

    def _build_mesh(self, min_angle):
        input_data = {
            'vertices': self.verts_full,
            'segments': self.segs,
            'segment_markers': np.ones(len(self.segs), dtype=int)
        }
        mesh = tr.triangulate(input_data, f'pq{min_angle:.1f}ez')

        self.verts = mesh['vertices'].astype(np.float32)
        self.triangles = mesh['triangles']
        self.n_verts = len(self.verts)

    def _extract_edges(self):
        edge_set = set()
        for tri in self.triangles:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i+1) % 3])
                edge_set.add((a, b) if a < b else (b, a))
        self.edges = list(edge_set)
        self.rest_lengths = np.array([
            np.linalg.norm(self.verts[i]-self.verts[j])
            for i, j in self.edges
        ], dtype=np.float32)

    def get_interior_indices(self):
        all_idx = np.arange(self.n_verts)
        return np.setdiff1d(all_idx, self.boundary_idx)
