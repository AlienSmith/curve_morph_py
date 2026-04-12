import numpy as np
from typing import List, Tuple
from .core import Particle, PBDSolver
from .constraints import DistanceConstraint, FourierBoundaryConstraint
from .mesh_builder import MeshBuilder
from .fourier_morpher import FourierShapeMorpher


def generate_morph(pts_A: np.ndarray, pts_B: np.ndarray, num_frames: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    morpher = FourierShapeMorpher(pts_A, pts_B)
    builder = MeshBuilder(pts_A, min_angle=30.0)
    particles = [Particle(pos) for pos in builder.verts]
    # Static elastic constraints
    static_constraints = [
        DistanceConstraint(i, j, rest_len, stiffness=0.1)
        for (i, j), rest_len in zip(builder.edges, builder.rest_lengths)
    ]

    frames = []
    for t in np.linspace(0, 1, num_frames):
        target = morpher.evaluate(t)

        # Soft boundary pull
        fourier_c = FourierBoundaryConstraint(
            builder.boundary_idx.tolist(), target, alpha=0.1
        )

        # ORDER MATTERS: Soft guide → Elastic smoothing
        all_c = [fourier_c] + static_constraints

        solver = PBDSolver(particles, all_c, dt=1.0, damping=0.98)
        pos = solver.step(iterations=200, tol=1e-5)
        frames.append(pos.copy())
        # print(
        #     f"  Frame {len(frames)}/{num_frames} | max_disp: {solver.iteration_log[-1] if solver.iteration_log else 0:.6f}")

    return np.array(frames), builder.boundary_idx
