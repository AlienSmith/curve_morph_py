import matplotlib.pyplot as plt
import numpy as np


def main():
    # Simulation parameters (same as original Mocz code)
    Nx = 400  # resolution x-dir
    Ny = 100  # resolution y-dir
    rho0 = 100  # average density
    tau = 0.6  # collision timescale
    Nt = 4000  # number of timesteps
    plotRealTime = True

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    opp = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])

    # --- Build MRT Matrix (same as corrected code above) ---
    m0 = np.ones(9)
    m1 = cxs.copy()
    m2 = cys.copy()
    m3 = cxs**2 - cys**2
    m4 = cxs * cys
    m5 = cxs**2 + cys**2
    m6 = cxs * (cxs**2 + cys**2)
    m7 = cys * (cxs**2 + cys**2)
    m8 = (cxs**2 + cys**2)**2
    raw_moments = np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8])

    T = np.zeros_like(raw_moments)
    for k in range(9):
        v = raw_moments[k].copy()
        for j in range(k):
            proj = np.dot(v * weights, T[j]) / np.dot(T[j] * weights, T[j])
            v -= proj * T[j]
        norm = np.sqrt(np.dot(v * weights, v))
        T[k, :] = v / norm if norm > 1e-12 else v
    T_inv = np.diag(weights) @ T.T
    s_v = 1.0 / tau
    s = np.array([0, 0, 0, s_v, s_v, 1.0, 1.2, 1.2, 1.0])

    # Initial Conditions (same as original)
    F = np.ones((Ny, Nx, NL))
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Cylinder boundary (same as original)
    cylinder = (X - Nx / 4) ** 2 + (Y - Ny / 2) ** 2 < (Ny / 4) ** 2

    # Prep figure
    fig = plt.figure(figsize=(4, 2), dpi=80)

    # Simulation Main Loop (keeps original streaming-first order)
    for it in range(Nt):
        print(it)

        # Drift / Streaming
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Save pre-collision boundary populations for bounce-back
        bndryF = F[cylinder, :][:, opp]

        # Calculate fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        # Compute equilibrium
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3*(cx*ux + cy*uy) + 9*(cx*ux + cy*uy)**2 /
                2 - 3*(ux**2 + uy**2)/2
            )

        # --- Replace BGK collision with MRT collision ---
        M = F @ T.T
        Meq = Feq @ T.T
        M -= s * (M - Meq)
        F = M @ T_inv.T

        # Apply boundary (overwrite collided cylinder populations with bounced pre-collision values)
        F[cylinder, :] = bndryF

        # Plotting (same as original)
        if (plotRealTime and (it % 10) == 0) or (it == Nt - 1):
            plt.cla()
            ux[cylinder] = 0
            uy[cylinder] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity[cylinder] = np.nan
            vorticity = np.ma.array(vorticity, mask=cylinder)
            plt.imshow(vorticity, cmap="bwr")
            plt.imshow(~cylinder, cmap="gray", alpha=0.3)
            plt.clim(-0.1, 0.1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)

    plt.savefig("latticeboltzmann_mrt.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
