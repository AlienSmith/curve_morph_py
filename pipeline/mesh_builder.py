import numpy as np
import triangle as tr
from typing import List, Tuple


class MeshBuilder:
    def __init__(self, pts_A: np.ndarray, min_angle: float = 30.0):
        """
        pts_A: (N, 2) array of points forming a closed loop.
        """
        self.A = np.asarray(pts_A, dtype=float)
        self.N = len(self.A)

        self._build_mesh(min_angle)
        self._extract_edges()

        # Since we pass the vertices first, triangle preserves their
        # indices (0 to N-1) as the boundary in the order provided.
        self.boundary_idx = np.arange(self.N)

    def _build_mesh(self, min_angle: float):
        # Create segments connecting point i to i+1 (and N-1 to 0)
        segs = np.stack(
            [np.arange(self.N), np.roll(np.arange(self.N), -1)], axis=1)

        # 'segment_markers' helps triangle identify which edges are boundaries
        input_data = {
            'vertices': self.A,
            'segments': segs,
            'segment_markers': np.ones(len(segs), dtype=int)
        }

        # p: Planar Straight Line Graph
        # q: Quality mesh (min_angle)
        # e: Generate edge list and markers
        # z: Start numbering from zero
        mesh = tr.triangulate(input_data, f'pq{min_angle:.1f}ez')

        self.verts = mesh['vertices'].astype(np.float32)
        self.triangles = mesh['triangles']
        self.n_verts = len(self.verts)

    def _extract_edges(self):
        """Extracts all unique edges from the triangulation for DistanceConstraints."""
        edge_set = set()
        for tri in self.triangles:
            # For each triangle, add its 3 edges
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i+1) % 3])
                # Store as sorted tuple to avoid (a,b) and (b,a) duplicates
                edge_set.add((a, b) if a < b else (b, a))

        self.edges = list(edge_set)

        # Pre-calculate rest lengths based on the initial shape (pts_A)
        self.rest_lengths = np.array([
            np.linalg.norm(self.verts[i] - self.verts[j])
            for i, j in self.edges
        ], dtype=np.float32)

    def get_interior_indices(self) -> np.ndarray:
        """Helper to get indices of vertices that are NOT on the boundary."""
        all_indices = np.arange(self.n_verts)
        # In this specific case, 0 to N-1 are boundary, N to end are interior
        return all_indices[self.N:]
