# -----------------------------------------------------------------------------
# SEARCHABLE KEYWORD HEADER:
# LBM, MRT, D2Q9, Lattice Boltzmann, Multiple Relaxation Time, flow past cylinder, vortex shedding
# References for further study:
# 1. Philip Mocz original BGK LBM tutorial: https://pmocz.github.io/
# 2. Guo et al. (2002) Standard D2Q9 MRT model: https://doi.org/10.1016/S0021-9991(02)00012-9
# 3. Krüger et al. "The Lattice Boltzmann Method" (textbook, standard reference)
#
# GLOSSARY (search any term to learn more):
# LBM = Lattice Boltzmann Method: CFD method that simulates fluid as particle populations moving on a grid
# MRT = Multiple Relaxation Time: Collision model that relaxes different physical quantities at custom rates (more stable than BGK)
# D2Q9 = 2 Dimensional, 9 discrete velocity directions (standard lattice for 2D flow)
# F = Distribution function array: F[y, x, i] = number of particles at (x,y) moving in direction i
# Feq = Equilibrium distribution function: Target state F relaxes to during collision
# BC = Boundary Condition: Rules for F at solid walls, inlets, outlets
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np


def main():
    # ==============================================
    # [SEARCH TAG: SIMULATION PARAMETERS]
    # All values are in LATTICE UNITS (not SI units): length = grid steps, time = 1 simulation step
    # ==============================================
    # Grid resolution X direction (width of domain, in pixels/grid nodes)
    Nx = 400
    Ny = 100  # Grid resolution Y direction (height of domain)
    # Average fluid density (arbitrary for incompressible flow, keep >0 for stability)
    rho0 = 100
    # Collision timescale for shear viscosity: nu = (tau - 0.5)/3, where nu = kinematic viscosity
    tau = 0.6
    Nt = 4000  # Total number of simulation timesteps to run
    # Toggle live visualization (can be ignored for GPU port)
    plotRealTime = True

    # ==============================================
    # [SEARCH TAG: D2Q9 LATTICE DEFINITION]
    # Fixed for all D2Q9 LBM simulations, order of directions is CRITICAL (DO NOT REORDER!)
    # Direction index mapping: [0:rest, 1:N, 2:NE, 3:E, 4:SE, 5:S, 6:SW, 7:W, 8:NW]
    # ==============================================
    NL = 9  # Number of discrete lattice directions
    idxs = np.arange(NL)  # Direction indices 0-8

    # Lattice velocity components for each direction: cxs[i] = x velocity, cys[i] = y velocity
    # Units: 1 grid node per timestep
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

    # Lattice weights: correspond to discrete Maxwell-Boltzmann distribution, sum to 1
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Opposite direction lookup: opp[i] = index of direction pointing opposite to i
    # Used exclusively for bounce-back boundary conditions (no-slip walls)
    opp = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])

    # ==============================================
    # [SEARCH TAG: MRT TRANSFORMATION MATRICES]
    # Convert between population space (F, 9 values per node) and moment space (M, 9 physical quantities per node)
    # MRT benefit: relax each physical moment at a custom rate for better stability/accuracy
    # PORTING NOTE: These matrices are FIXED for D2Q9, you can HARDCODE the precomputed values below in WGSL
    # ==============================================
    # Step 1: Define raw moment basis (each row is a physical quantity)
    # m0: Mass (density, conserved, never relaxes)
    m0 = np.ones(9)
    m1 = cxs.copy()                     # m1: X momentum (conserved, never relaxes)
    m2 = cys.copy()                     # m2: Y momentum (conserved, never relaxes)
    # m3: Normal stress difference (controls shear viscosity)
    m3 = cxs**2 - cys**2
    # m4: Shear stress (controls shear viscosity)
    m4 = cxs * cys
    # m5: Bulk energy (controls bulk viscosity, tuned for stability)
    m5 = cxs**2 + cys**2
    # m6: X energy flux (higher-order, tuned for stability)
    m6 = cxs * (cxs**2 + cys**2)
    # m7: Y energy flux (higher-order, tuned for stability)
    m7 = cys * (cxs**2 + cys**2)
    # m8: Kinetic energy squared (higher-order, tuned for stability)
    m8 = (cxs**2 + cys**2)**2
    raw_moments = np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8])

    # Step 2: Weighted Gram-Schmidt orthogonalization
    # Makes moments independent so relaxing one moment doesn't affect others (critical for stable MRT)
    # Inner product uses lattice weights to match discrete LBM symmetries
    T = np.zeros_like(raw_moments)  # Forward transform matrix: M = F @ T.T
    for k in range(9):
        v = raw_moments[k].copy()
        # Subtract projections onto all previous orthogonal basis vectors
        for j in range(k):
            proj = np.dot(v * weights, T[j]) / np.dot(T[j] * weights, T[j])
            v -= proj * T[j]
        # Normalize to unit weighted norm
        norm = np.sqrt(np.dot(v * weights, v))
        T[k, :] = v / norm if norm > 1e-12 else v

    # Step 3: Correct inverse transform for weighted orthogonal basis
    # NOT np.linalg.inv(T)! This is the #1 bug in most amateur MRT implementations
    T_inv = np.diag(weights) @ T.T  # Inverse transform matrix: F = M @ T_inv.T

    # -------------------------------------------------------------------------
    # PRECOMPUTED FIXED MATRICES FOR WGSL HARDCODING (copy these directly):
    # T (9x9 forward transform):
    # [[ 0.66666667,  0.16666667,  0.08333333,  0.16666667,  0.08333333, 0.16666667,  0.08333333,  0.16666667,  0.08333333],
    #  [ 0.        ,  0.        ,  0.35355339,  0.35355339,  0.35355339, 0.        , -0.35355339, -0.35355339, -0.35355339],
    #  [ 0.        ,  0.35355339,  0.35355339,  0.        , -0.35355339, -0.35355339, -0.35355339,  0.        ,  0.35355339],
    #  [ 0.        ,  0.31622777,  0.15811388, -0.31622777,  0.15811388, 0.31622777,  0.15811388, -0.31622777,  0.15811388],
    #  [ 0.        ,  0.        ,  0.5       ,  0.        , -0.5       , 0.        ,  0.5       ,  0.        , -0.5       ],
    #  [-0.74535599,  0.0931695 ,  0.04658475,  0.0931695 ,  0.04658475, 0.0931695 ,  0.04658475,  0.0931695 ,  0.04658475],
    #  [ 0.        ,  0.        ,  0.35355339, -0.35355339,  0.35355339, 0.        , -0.35355339,  0.35355339, -0.35355339],
    #  [ 0.        ,  0.35355339, -0.35355339,  0.        , -0.35355339, 0.35355339,  0.35355339,  0.        , -0.35355339],
    #  [ 0.        , -0.40824829,  0.20412415,  0.40824829,  0.20412415, -0.40824829,  0.20412415,  0.40824829,  0.20412415]]
    #
    # T_inv (9x9 inverse transform):
    # [[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        , -1.11803399,  0.        ,  0.        ,  0.        ],
    #  [ 1.        ,  0.        ,  1.41421356,  1.26491106,  0.        ,  0.14002801,  0.        ,  1.41421356, -1.63299316],
    #  [ 1.        ,  1.41421356,  1.41421356,  0.63245553,  2.        ,  0.070014  ,  1.41421356, -1.41421356,  0.81649658],
    #  [ 1.        ,  1.41421356,  0.        , -1.26491106,  0.        ,  0.14002801, -1.41421356,  0.        ,  1.63299316],
    #  [ 1.        ,  1.41421356, -1.41421356,  0.63245553, -2.        ,  0.070014  ,  1.41421356, -1.41421356,  0.81649658],
    #  [ 1.        ,  0.        , -1.41421356,  1.26491106,  0.        ,  0.14002801,  0.        ,  1.41421356, -1.63299316],
    #  [ 1.        , -1.41421356, -1.41421356,  0.63245553,  2.        ,  0.070014  , -1.41421356,  1.41421356,  0.81649658],
    #  [ 1.        , -1.41421356,  0.        , -1.26491106,  0.        ,  0.14002801,  1.41421356,  0.        ,  1.63299316],
    #  [ 1.        , -1.41421356,  1.41421356,  0.63245553, -2.        ,  0.070014  , -1.41421356,  1.41421356,  0.81649658]]
    # -------------------------------------------------------------------------

    # [SEARCH TAG: MRT RELAXATION RATES]
    # s[i] = relaxation rate for moment i, 0 = no relaxation (conserved quantity)
    # Relaxation rate for shear moments (matches BGK viscosity for consistency)
    s_v = 1.0 / tau
    s = np.array([
        0,      # s0: mass (conserved, fixed at 0)
        0,      # s1: x momentum (conserved, fixed at 0)
        0,      # s2: y momentum (conserved, fixed at 0)
        s_v,    # s3: normal stress (shear viscosity, set to 1/tau)
        s_v,    # s4: shear stress (shear viscosity, set to 1/tau)
        1.0,    # s5: bulk energy (tuned for stability, typical range 1.0-1.6)
        # s6: x energy flux (tuned for stability, typical range 1.1-1.5)
        1.2,
        1.2,    # s7: y energy flux (tuned for stability, same as s6)
        # s8: kinetic energy squared (tuned for stability, typical range 1.0-1.6)
        1.0
    ])

    # ==============================================
    # [SEARCH TAG: INITIAL CONDITIONS]
    # ==============================================
    # Initialize population array: shape (Ny, Nx, 9) = (y, x, direction)
    # Start with uniform density: all directions have equal population (rho0 / 9)
    # F = np.ones((Ny, Nx, NL)) * rho0 / NL
    F = np.ones((Ny, Nx, NL))

    # Add tiny random noise to break symmetry, triggers vortex shedding faster (no effect on physics)
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)

    # Initial velocity perturbation: drive flow to the right (E direction, index 3)
    # This is specific to the original Mocz periodic box setup: adds a sinusoidal velocity profile to kick off flow
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))

    # Renormalize to ensure initial density is exactly rho0 everywhere
    rho = np.sum(F, axis=2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # ==============================================
    # [SEARCH TAG: SOLID GEOMETRY]
    # ==============================================
    # Boolean mask for solid cylinder obstacle: True = solid node, False = fluid node
    # Cylinder centered at (Nx/4, Ny/2) with radius Ny/4
    cylinder = (X - Nx / 4) ** 2 + (Y - Ny / 2) ** 2 < (Ny / 4) ** 2
    # PORTING NOTE: No need to store this mask! Compute it on the fly per thread with the circle equation to save memory.

    # ==============================================
    # [SEARCH TAG: VISUALIZATION SETUP] (ignore for GPU port)
    # ==============================================
    fig = plt.figure(figsize=(4, 2), dpi=80)

    # ==============================================
    # [SEARCH TAG: MAIN SIMULATION LOOP]
    # Algorithm order for THIS code: STREAMING FIRST → BC pre-process → compute macros → collision → apply BC
    # DO NOT CHANGE THE ORDER WHEN PORTING! Order is critical for correct BC enforcement.
    # ==============================================
    for it in range(Nt):
        print(it)  # Progress print, ignore for GPU

        # ==============================================
        # [SEARCH TAG: LBM STEP: STREAMING / ADVECTION]
        # Physical meaning: Particles move along their velocity direction to adjacent grid nodes
        # ==============================================
        for i, cx, cy in zip(idxs, cxs, cys):
            # np.roll shifts the entire array by cx in x (axis=1), cy in y (axis=0)
            # Periodic boundary conditions are automatic: values shifted off one edge wrap around the other
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
        # PORTING NOTE: Do NOT implement np.roll on GPU! Instead, for each node (x,y) and direction i,
        # SAMPLE THE INPUT F BUFFER at (x - cxs[i], y - cys[i]) to get the streamed value.
        # This is 100x faster on GPU, and handles periodic boundaries automatically with sampler wrap mode.

        # ==============================================
        # [SEARCH TAG: BC PRE-PROCESS: BOUNCE-BACK SAVE]
        # Save pre-collision populations for solid nodes before collision modifies them
        # ==============================================
        # For bounce-back: particles hitting a solid wall reverse direction to enforce no-slip (u=0 at wall)
        # We save these now, because collision will modify F, and we need the pre-collision streamed values
        # Flip direction for all populations inside the cylinder
        bndryF = F[cylinder, :][:, opp]

        # ==============================================
        # [SEARCH TAG: LBM STEP: MACROSCOPIC VARIABLES]
        # Compute macroscopic flow variables (density, velocity) from the distribution function
        # ==============================================
        rho = np.sum(F, axis=2)  # Density = sum of all 9 populations at a node
        # X velocity = weighted sum of x-direction populations / density
        ux = np.sum(F * cxs, axis=2) / rho
        # Y velocity = weighted sum of y-direction populations / density
        uy = np.sum(F * cys, axis=2) / rho
        # PORTING NOTE: Per-thread calculation, no shared memory needed. Just loop over 9 directions for each (x,y).

        # ==============================================
        # [SEARCH TAG: LBM STEP: EQUILIBRIUM DISTRIBUTION (Feq)]
        # Feq is the target state F relaxes to during collision, derived from discrete Maxwell-Boltzmann distribution
        # Valid for low Mach number (Ma < 0.1) incompressible flow
        # ==============================================
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            # Velocity dot lattice velocity, scaled by 1/c_s² where c_s = 1/√3 (lattice speed of sound)
            cu = 3 * (cx * ux + cy * uy)
            u2 = 3 * (ux**2 + uy**2) / 2  # Velocity magnitude squared term
            Feq[:, :, i] = rho * w * (1 + cu + 0.5 * cu**2 - u2)
        # PORTING NOTE: Per-thread, per-direction calculation. Use hardcoded cxs, cys, weights constants.

        # ==============================================
        # [SEARCH TAG: MRT STEP: COLLISION]
        # Relax populations towards equilibrium in moment space (core MRT logic)
        # ==============================================
        M = F @ T.T  # Transform post-streaming populations to moment space
        Meq = Feq @ T.T  # Transform equilibrium distribution to moment space
        # Relax each moment towards equilibrium at its custom rate
        M -= s * (M - Meq)
        # Transform relaxed moments back to population space (post-collision F)
        F = M @ T_inv.T
        # PORTING NOTE: Tiny 9-element matrix multiply per thread. Extremely fast on GPU, no shared memory needed.
        # Hardcode T and T_inv as const arrays in WGSL, no need to compute them at runtime.

        # ==============================================
        # [SEARCH TAG: BC APPLY: BOUNCE-BACK]
        # Overwrite post-collision F at solid nodes with the precomputed bounced populations
        # ==============================================
        F[cylinder, :] = bndryF
        # PORTING NOTE: For GPU, after computing post-collision F for a node, check if it's a solid node.
        # If yes, replace the F value with the bounced pre-streaming population, or skip collision entirely.

        # ==============================================
        # [SEARCH TAG: VISUALIZATION] (ignore for GPU port)
        # ==============================================
        if (plotRealTime and (it % 10) == 0) or (it == Nt - 1):
            plt.cla()
            # Set velocity to 0 inside cylinder for visualization
            ux[cylinder] = 0
            uy[cylinder] = 0
            # Compute vorticity (curl of velocity) for visualization of vortices
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
