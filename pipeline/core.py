from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Callable


class Particle:
    def __init__(self, pos: np.ndarray, inv_mass: float = 1.0):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.prev_pos = self.pos.copy()
        self.inv_mass = inv_mass


class Constraint(ABC):
    @abstractmethod
    def resolve(self, particles: List[Particle]) -> float:
        """Returns max displacement for convergence tracking."""
        pass


class PBDSolver:
    def __init__(self, particles: List[Particle], constraints: List[Constraint],
                 dt: float = 1.0, damping: float = 0.98):
        self.particles = particles
        self.constraints = constraints
        self.dt = dt
        self.damping = damping

    def step(self, iterations: int = 200, tol: float = 1e-5) -> np.ndarray:
        # 1. Predict
        for p in self.particles:
            p.prev_pos = p.pos.copy()
            p.pos += p.vel * self.dt

        # 2. Solve
        for _ in range(iterations):
            max_disp = 0.0
            for c in self.constraints:
                d = c.resolve(self.particles)
                if d > max_disp:
                    max_disp = d
            if max_disp < tol:
                break

        # 3. Update velocity
        for p in self.particles:
            p.vel = (p.pos - p.prev_pos) / self.dt
            p.vel *= self.damping

        return np.array([p.pos for p in self.particles])
