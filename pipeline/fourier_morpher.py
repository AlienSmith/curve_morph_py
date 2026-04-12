import numpy as np

class FourierShapeMorpher:
    def __init__(self, pts_A, pts_B):
        self.A = pts_A
        self.B = pts_B
        self.n_points = len(pts_A)
        self._precompute()

    def _precompute(self):
        N = self.n_points
        Az = self.A[:, 0] + 1j * self.A[:, 1]
        Bz = self.B[:, 0] + 1j * self.B[:, 1]
        self.centroid_A = Az.mean()
        self.centroid_B = Bz.mean()
        A_c = Az - self.centroid_A
        B_c = Bz - self.centroid_B
        self.scale = np.linalg.norm(A_c) / (np.linalg.norm(B_c) + 1e-8)
        B_c *= self.scale
        best_err, best_k = np.inf, 0
        for k in range(N):
            err = np.sum(np.abs(A_c - np.roll(B_c, k))**2)
            if err < best_err: best_err, best_k = err, k
        B_c = np.roll(B_c, best_k)
        self.FA = np.fft.fft(A_c)
        self.FB = np.fft.fft(B_c)
        self.phase_shift = best_k

    def evaluate(self, t_ease):
        Ft = (1.0 - t_ease) * self.FA + t_ease * self.FB
        z_t = np.fft.ifft(Ft)
        centroid = (1.0 - t_ease) * self.centroid_A + t_ease * self.centroid_B
        z_t += centroid
        return np.c_[z_t.real.astype(np.float32), z_t.imag.astype(np.float32)]
