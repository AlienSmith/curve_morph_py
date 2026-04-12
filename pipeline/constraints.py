import time
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

        d21 = p2.pos - p1.pos
        d31 = p3.pos - p1.pos
        area_2 = d21[0] * d31[1] - d21[1] * d31[0]

        # Prevent zero / negative area
        min_area2 = 1e-6 * abs(2 * self.rest_area)
        area_2 = np.clip(area_2, min_area2, None)

        C = 0.5 * area_2 - self.rest_area
        if abs(C) < 1e-6:
            return 0.0

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
        self.boundary = boundary_indices
        self.thickness = thickness
        # Precompute boundary edges ONCE, not every resolve
        self.edges = [
            (boundary_indices[k], boundary_indices[(k+1) %
             len(boundary_indices)])
            for k in range(len(boundary_indices))
        ]

    def resolve(self, particles: List[Particle]) -> float:
        max_corr = 0.0

        # Only check BOUNDARY POINTS against BOUNDARY EDGES
        # (interior vertices are safe if area constraints are strong)
        for vi in self.boundary:
            p = particles[vi]
            for (a_idx, b_idx) in self.edges:
                # Skip point's own two adjacent edges
                if vi == a_idx or vi == b_idx:
                    continue

                A = particles[a_idx].pos
                B = particles[b_idx].pos

                edge = B - A
                L2 = np.dot(edge, edge)
                if L2 < 1e-12:
                    continue

                # Closest point on segment
                t = np.dot(p.pos - A, edge) / L2
                t = np.clip(t, 0.0, 1.0)
                proj = A + t * edge

                diff = p.pos - proj
                dist = np.linalg.norm(diff)

                if dist < self.thickness:
                    # Push outward only (simple, stable)
                    corr = (self.thickness - dist) * 0.3
                    normal = diff / (dist + 1e-12)
                    p.pos += corr * normal
                    if corr > max_corr:
                        max_corr = corr

        return max_corr


class TimeLogConstraint(Constraint):
    def __init__(self, name: str):
        self.name = name

    def resolve(self, particles):
        now = time.perf_counter()
        print(f"[TIMER] {self.name} | resolve() called at {now:.6f}s")
        return 0.0
