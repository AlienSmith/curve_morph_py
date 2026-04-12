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
