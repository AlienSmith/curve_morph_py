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

        # --------------------------
        # CRITICAL: ENFORCE ORIGINAL SIGN
        # --------------------------
        target_sign = np.sign(self.rest_area)
        current_sign = np.sign(area_2)

        # If sign flipped, force it back (prevents inversion)
        if current_sign != target_sign:
            area_2 = 1e-6 * target_sign * abs(2 * self.rest_area)

        # Keep area magnitude reasonable
        min_area2 = 1e-6 * abs(2 * self.rest_area)
        if abs(area_2) < min_area2:
            area_2 = min_area2 * target_sign

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
    def __init__(self, boundary_indices: list, thickness: float = 0.03, skip_neighbors: int = 12):
        self.b_verts = boundary_indices
        self.n = len(boundary_indices)
        self.thick = thickness
        self.skip = skip_neighbors  # Skip nearby edges (CANNOT collide)

        # Precompute ALL boundary edges ONCE
        self.edges = [
            (boundary_indices[i], boundary_indices[(i+1) % self.n])
            for i in range(self.n)
        ]

    def resolve(self, particles):
        max_corr = 0.0
        n = self.n
        skip = self.skip

        # Iterate each boundary point
        for i in range(n):
            vid = self.b_verts[i]
            p = particles[vid].pos

            # ONLY CHECK DISTANT EDGES — SKIP SELF + NEIGHBORS (HUGE SPEEDUP)
            for j in range(i + skip, i + n - skip):
                j %= n
                a_id, b_id = self.edges[j]

                # Skip edges connected to current vertex
                if vid == a_id or vid == b_id:
                    continue

                A = particles[a_id].pos
                B = particles[b_id].pos
                dx, dy = B - A

                # Vector math (optimized, no norm until needed)
                px = p[0] - A[0]
                py = p[1] - A[1]
                dot = px * dx + py * dy
                len2 = dx * dx + dy * dy

                if len2 < 1e-12:
                    continue

                t = dot / len2
                t = max(0.0, min(1.0, t))
                cx = A[0] + t * dx
                cy = A[1] + t * dy

                # Distance
                d_x = p[0] - cx
                d_y = p[1] - cy
                dist = np.hypot(d_x, d_y)

                if dist < self.thick:
                    # Push out
                    scale = (self.thick - dist) * 0.4 / (dist + 1e-8)
                    particles[vid].pos[0] += d_x * scale
                    particles[vid].pos[1] += d_y * scale
                    corr = abs(self.thick - dist)
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
