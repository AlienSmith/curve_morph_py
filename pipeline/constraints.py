import numpy as np
from typing import List
from .core import Particle, Constraint


class DistanceConstraint(Constraint):
    def __init__(self, i: int, j: int, rest_length: float, stiffness: float = 1.0):
        self.i, self.j = i, j
        self.rest_length = rest_length
        self.stiffness = stiffness

    def resolve(self, particles: List[Particle]) -> float:
        p1, p2 = particles[self.i], particles[self.j]
        delta = p1.pos - p2.pos
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return 0.0

        n = delta / dist
        w_sum = p1.inv_mass + p2.inv_mass
        correction = (dist - self.rest_length) / w_sum * self.stiffness

        p1.pos -= correction * p1.inv_mass * n
        p2.pos += correction * p2.inv_mass * n
        return abs(correction) * max(p1.inv_mass, p2.inv_mass)


class FourierBoundaryConstraint(Constraint):
    def __init__(self, boundary_indices: List[int], target: np.ndarray, alpha: float = 0.04):
        self.indices = boundary_indices
        self.target = np.asarray(target, dtype=float)
        self.alpha = alpha

    def resolve(self, particles: List[Particle]) -> float:
        max_disp = 0.0
        for k, idx in enumerate(self.indices):
            diff = self.target[k] - particles[idx].pos
            corr = self.alpha * diff
            particles[idx].pos += corr
            d = np.linalg.norm(corr)
            if d > max_disp:
                max_disp = d
        return max_disp


class TriangleAreaConstraint(Constraint):
    def __init__(self, i, j, k, rest_area, stiffness=0.1):
        self.indices = [i, j, k]
        self.rest_area = rest_area
        self.stiffness = stiffness

    def resolve(self, particles: List[Particle]) -> float:
        p1, p2, p3 = [particles[i] for i in self.indices]

        # Current signed area * 2
        d21 = p2.pos - p1.pos
        d31 = p3.pos - p1.pos
        area_2 = d21[0] * d31[1] - d21[1] * d31[0]

        C = 0.5 * area_2 - self.rest_area
        if abs(C) < 1e-6:
            return 0.0

        # Gradients (grad C w.r.t p1, p2, p3)
        grad1 = 0.5 * np.array([p2.pos[1] - p3.pos[1], p3.pos[0] - p2.pos[0]])
        grad2 = 0.5 * np.array([p3.pos[1] - p1.pos[1], p1.pos[0] - p3.pos[0]])
        grad3 = 0.5 * np.array([p1.pos[1] - p2.pos[1], p2.pos[0] - p1.pos[0]])

        w_sum = (p1.inv_mass * np.sum(grad1**2) +
                 p2.inv_mass * np.sum(grad2**2) +
                 p3.inv_mass * np.sum(grad3**2))

        if w_sum < 1e-9:
            return 0.0

        s = C / w_sum * self.stiffness
        p1.pos -= s * p1.inv_mass * grad1
        p2.pos -= s * p2.inv_mass * grad2
        p3.pos -= s * p3.inv_mass * grad3
        return abs(s)


class SelfCollisionConstraint(Constraint):
    def __init__(self, boundary_indices: List[int], thickness: float = 0.02):
        self.indices = boundary_indices
        self.thickness = thickness

    def resolve(self, particles: List[Particle]) -> float:
        max_corr = 0.0
        # Simple N^2 check for demonstration; for production use Spatial Hashing
        for i in self.indices:
            p = particles[i]
            for idx in range(len(self.indices)):
                # Define edge (A, B)
                a_idx = self.indices[idx]
                b_idx = self.indices[(idx + 1) % len(self.indices)]

                # Skip if vertex is part of the edge
                if i == a_idx or i == b_idx:
                    continue

                A, B = particles[a_idx].pos, particles[b_idx].pos
                edge = B - A
                L2 = np.sum(edge**2)
                if L2 < 1e-9:
                    continue

                # Project p onto edge AB
                t = max(0, min(1, np.dot(p.pos - A, edge) / L2))
                projection = A + t * edge
                diff = p.pos - projection
                dist = np.linalg.norm(diff)

                if dist < self.thickness:
                    # Resolve collision
                    normal = diff / \
                        (dist + 1e-9) if dist > 1e-9 else np.array([0, 1])
                    corr = (self.thickness - dist) * 0.5
                    p.pos += corr * normal
                    # (In a full solver, you'd move A and B back too)
                    max_corr = max(max_corr, corr)
        return max_corr
