# main.py
import numpy as np
import matplotlib.pyplot as plt
from loader import load_and_upgrade
from obb import OBBAligner
from renderer import SDFRenderer
from sampler import ContourSampler
from utils import reconstruct_bezier_controls
# ---------------------------------------------------------
# DEBUG VISUALIZATION HELPER
# ---------------------------------------------------------


def evaluate_curve_for_plot(pts: np.ndarray, samples_per_seg: int = 64) -> np.ndarray:
    """Vectorized high-res curve evaluation for plotting."""
    N = len(pts) // 2
    t = np.linspace(0, 1, samples_per_seg, endpoint=False)
    t2 = t[:, np.newaxis]

    # Quadratic Bézier: (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    P0 = pts[0::2][np.newaxis, :, :]  # (1, N, 2)
    P1 = pts[1::2][np.newaxis, :, :]  # (1, N, 2)
    P2 = pts[0::2][np.newaxis, :, :]  # will roll later
    P2 = np.roll(P2, -1, axis=1)      # wrap to next anchor

    curve = (1 - t2)**2 * P0 + 2 * (1 - t2) * t2 * P1 + t2**2 * P2
    return curve.reshape(-1, 2)


def debug_plot_input(pts_a: np.ndarray, pts_b: np.ndarray, title: str = "Debug: Loaded Shapes"):
    """Plot two [A,C,A,C...] quad Bézier chains with anchors & controls."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for label, color, pts in [("Shape A", "#1f77b4", pts_a), ("Shape B", "#d62728", pts_b)]:
        anchors = pts[0::2]
        controls = pts[1::2]
        curve = evaluate_curve_for_plot(pts)

        ax.plot(curve[:, 0], curve[:, 1],
                color=color, linewidth=2, label=label)
        ax.plot(anchors[:, 0], anchors[:, 1], 'o',
                color=color, markersize=4, alpha=0.8)
        ax.plot(controls[:, 0], controls[:, 1], 'x',
                color=color, markersize=5, alpha=0.8)

    ax.legend()
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# PIPELINE ENTRY
# ---------------------------------------------------------
TARGET_SEGMENTS = 64
DEBUG = True  # Toggle visualization


def debug_pipeline(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if DEBUG:
        debug_plot_input(
            A, B, f"Loaded & Upgraded Curves (Segments={TARGET_SEGMENTS})")

    aligner = OBBAligner(A, B)

    # Quick verification
    print(
        f"📐 OBB A: center={aligner.obb_a.center.round(2)}, half_size={aligner.obb_a.half_size.round(2)}")
    print(
        f"📐 OBB B: center={aligner.obb_b.center.round(2)}, half_size={aligner.obb_b.half_size.round(2)}")

    # Test canonical mapping
    A_can = aligner.to_canonical(A, 'a')
    B_can = aligner.to_canonical(B, 'b')

    # Test world interpolation at α=0.5
    test_pts = A_can[0::2]  # Use anchors for quick visual check
    morphed = aligner.to_world(0.5, test_pts)

    if DEBUG:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax.plot(A[0::2, 0], A[0::2, 1], 'b--', alpha=0.6, label='A World')
        ax.plot(B[0::2, 0], B[0::2, 1], 'r--', alpha=0.6, label='B World')
        ax.plot(A_can[0::2, 0], A_can[0::2, 1], 'c-', label='A Canonical')
        ax.plot(B_can[0::2, 0], B_can[0::2, 1], 'm-', label='B Canonical')
        ax.plot(morphed[:, 0], morphed[:, 1], 'go',
                label=f'Morphed Anchors (α=0.5)')

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.legend()
        ax.set_title("Rotating Calipers OBB Alignment")
        plt.tight_layout()
        plt.show()

    # Initialize renderer
    renderer = SDFRenderer(A_can, B_can, grid_size=256, padding_ratio=0.15)

    # Generate interpolated SDF at alpha=0.5
    sdf = renderer.render(alpha)

    if DEBUG:

        # 1. Build world-space coordinate grids
        x_vals = np.linspace(
            renderer.x_min, renderer.x_max, renderer.grid_size)
        y_vals = np.linspace(
            renderer.y_min, renderer.y_max, renderer.grid_size)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 2. Plot SDF heatmap (flip vertically to match origin="lower")
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(sdf[::-1], cmap="coolwarm", origin="lower",
                       extent=[renderer.x_min, renderer.x_max, renderer.y_min, renderer.y_max])
        fig.colorbar(im, ax=ax, label="Signed Distance (world units)")

        # 3. Plot zero-level contour on correct world coordinates
        ax.contour(X, Y, sdf, levels=[0.0], colors="white", linewidths=2)

        # 4. Overlay canonical shapes for reference
        ax.plot(A_can[0::2, 0], A_can[0::2, 1],
                'c--', alpha=0.6, label='A Anchors')
        ax.plot(B_can[0::2, 0], B_can[0::2, 1],
                'm--', alpha=0.6, label='B Anchors')

        ax.set_xlim(renderer.x_min, renderer.x_max)
        ax.set_ylim(renderer.y_min, renderer.y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f"Interpolated SDF (α={alpha})")
        plt.tight_layout()
        plt.show()

        # Quick sanity print
        meta = renderer.get_grid_metadata()
        print(f"📐 Grid: {meta['grid_size']}×{meta['grid_size']}")
        print(
            f"🌍 Bounds: X[{meta['x_min']:.2f}, {meta['x_max']:.2f}], Y[{meta['y_min']:.2f}, {meta['y_max']:.2f}]")
        print(f"📏 Pixel Scale: {meta['pixel_scale']:.4f} units/px")

    sampler = ContourSampler(sdf, renderer.get_grid_metadata())
    TARGET_POINTS = 128
    canonical_pts = sampler.sample_boundary(TARGET_POINTS)

    if DEBUG:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # SDF heatmap
        im = ax.imshow(sdf[::-1], cmap="coolwarm", origin="lower",
                       extent=[renderer.x_min, renderer.x_max, renderer.y_min, renderer.y_max], alpha=0.7)
        fig.colorbar(im, ax=ax, label="Signed Distance")

        # Resampled points
        ax.plot(canonical_pts[:, 0], canonical_pts[:, 1], 'wo',
                markersize=4, label=f'Sampled ({TARGET_POINTS} pts)')

        # Highlight start point (+X axis intersection)
        ax.plot(canonical_pts[0, 0], canonical_pts[0, 1],
                'y*', markersize=15, label='Start Point (+X)')

        # Draw +X reference line
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
        ax.arrow(0, 0, 1.5, 0, head_width=0.1,
                 head_length=0.1, fc='r', ec='r', alpha=0.5)

        ax.set_xlim(renderer.x_min, renderer.x_max)
        ax.set_ylim(renderer.y_min, renderer.y_max)
        ax.legend(loc='upper right')
        ax.set_title(f"Canonical Boundary Sampling (α={alpha})")
        plt.tight_layout()
        plt.show()

        print(f"📐 Sampled shape: {canonical_pts.shape}")
        print(
            f"🎯 Start point: ({canonical_pts[0,0]:.3f}, {canonical_pts[0,1]:.3f})")
        print(
            f"📏 Perimeter: {np.sum(np.linalg.norm(np.diff(np.vstack([canonical_pts, canonical_pts[0]]), axis=0), axis=1)):.3f}")

    world_pts = aligner.to_world(alpha, canonical_pts)
    morphed_controls = reconstruct_bezier_controls(
        world_pts, anchors_at_odd_indices=True)  # Matches your request
    morphed_curve = evaluate_curve_for_plot(morphed_controls)

    if DEBUG:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Original shapes (dashed)
        ax.plot(A[0::2, 0], A[0::2, 1], 'b--', alpha=0.5, label='Shape A')
        ax.plot(B[0::2, 0], B[0::2, 1], 'r--', alpha=0.5, label='Shape B')

        # Morphed result (solid green)
        ax.plot(morphed_curve[:, 0], morphed_curve[:, 1],
                'g-', linewidth=2.5, label=f'Morphed (α={alpha})')

        # Reconstructed control structure
        ax.plot(morphed_controls[1::2, 0], morphed_controls[1::2,
                1], 'ko', markersize=4, label='Anchors (odd idx)')
        ax.plot(morphed_controls[0::2, 0], morphed_controls[0::2, 1],
                'rx', markersize=4, label='Controls (even idx)')

        ax.legend(loc='upper right')
        ax.set_title("Final Morphed Quad Bézier in World Space")
        plt.tight_layout()
        plt.show()

        print(f"✅ Pipeline complete. Output shape: {morphed_controls.shape}")
        print(f"📐 World bounds: X[{morphed_controls[:,0].min():.2f}, {morphed_controls[:,0].max():.2f}], "
              f"Y[{morphed_controls[:,1].min():.2f}, {morphed_controls[:,1].max():.2f}]")


if __name__ == "__main__":
    print("📥 Loading & upgrading shapes...")
    A = load_and_upgrade("/workspace/C_shape.json", TARGET_SEGMENTS)
    B = load_and_upgrade("/workspace/circle.json", TARGET_SEGMENTS)
    print(f"✅ Loaded A: {A.shape}, B: {B.shape}")
    debug_pipeline(A, B, 0.1)
