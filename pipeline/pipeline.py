import numpy as np
from typing import List, Tuple
from .core import Particle, PBDSolver
from .constraints import DistanceConstraint, FourierBoundaryConstraint, SelfCollisionConstraint, TimeLogConstraint, TriangleAreaConstraint
from .mesh_builder import MeshBuilder
from .fourier_morpher import FourierShapeMorpher


def generate_morph(pts_A: np.ndarray, pts_B: np.ndarray, num_frames: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    morpher = FourierShapeMorpher(pts_A, pts_B)
    builder = MeshBuilder(pts_A, min_angle=30.0)
    particles = [Particle(pos) for pos in builder.verts]

    # Static constraints (built ONCE)
    static_constraints = [
        DistanceConstraint(i, j, rest_len, stiffness=0.1)
        for (i, j), rest_len in zip(builder.edges, builder.rest_lengths)
    ]

    area_constraints = []
    for tri in builder.triangles:
        pts = builder.verts[tri]
        d21 = pts[1] - pts[0]
        d31 = pts[2] - pts[0]
        rest_area = 0.5 * (d21[0]*d31[1] - d21[1]*d31[0])
        area_constraints.append(TriangleAreaConstraint(
            tri[0], tri[1], tri[2], rest_area, stiffness=0.99))

    # collision_c = SelfCollisionConstraint(
    #     builder.boundary_idx.tolist(), thickness=0.03)

    # Solver created ONCE
    solver = PBDSolver(particles, [], dt=1.0, damping=0.98)

    frames = []
    for t in np.linspace(0, 1, num_frames):
        print(f"t = {t:.2f}")
        target = morpher.evaluate(t)

        # Only change the boundary constraint
        fourier_c = FourierBoundaryConstraint(
            builder.boundary_idx.tolist(), target, alpha=0.82
        )
        solver.constraints = [
            TimeLogConstraint("START"),
            fourier_c,
            TimeLogConstraint("AFTER_BOUNDARY"),

            TimeLogConstraint("START_DISTANCE"),
            *static_constraints,
            TimeLogConstraint("END_DISTANCE"),

            TimeLogConstraint("START_AREA"),
            *area_constraints,
            TimeLogConstraint("END_AREA"),

            TimeLogConstraint("START_COLLISION"),
            # collision_c,
            TimeLogConstraint("END_COLLISION"),
        ]

        solver.constraints = [
            fourier_c,
            *static_constraints,
            *area_constraints,
            # collision_c,
        ]

        # 200 iterations is way too many — we lower later, for now keep
        pos = solver.step(iterations=20, tol=1e-5)
        frames.append(pos.copy())

    return np.array(frames), builder.boundary_idx
