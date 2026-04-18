import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from loader import load_and_upgrade
from obb import OBBAligner
from renderer import SDFRenderer
from sampler import ContourSampler
from utils import evaluate_bezier_curve


def generate_morph_sequence(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    target_segments: int,
    num_frames: int = 10
) -> np.ndarray:
    """
    Generates a sequence of morphed quad Bézier control points.
    Returns shape: (num_frames, 2*target_segments, 2)
    Format per frame: [Control, Anchor, Control, Anchor, ...]
    """
    aligner = OBBAligner(pts_a, pts_b)
    A_can = aligner.to_canonical(pts_a, 'a')
    B_can = aligner.to_canonical(pts_b, 'b')

    renderer = SDFRenderer(A_can, B_can, grid_size=512, padding_ratio=0.15)
    metadata = renderer.get_grid_metadata()
    n_pts = target_segments * 2

    alphas = np.linspace(0.0, 1.0, num_frames)
    sequence = np.zeros((num_frames, n_pts, 2), dtype=np.float32)

    for i, alpha in enumerate(alphas):
        sdf = renderer.render(alpha)
        canonical_pts = ContourSampler(sdf, metadata).sample_boundary(n_pts)
        world_pts = aligner.to_world(alpha, canonical_pts)

        # Interleave: even indices → Controls, odd indices → Anchors
        sequence[i, 0::2] = world_pts[1::2]
        sequence[i, 1::2] = world_pts[0::2]

    return sequence


if __name__ == "__main__":
    TARGET_SEGMENTS = 64
    NUM_FRAMES = 120  # 61 frames → smooth ~2s GIF at 30fps
    GIF_PATH = "/workspace/morph_preview.gif"
    JSON_PATH = "/workspace/morph_output.json"

    print("📥 Loading & upgrading shapes...")
    A = load_and_upgrade("/workspace/C_shape.json", TARGET_SEGMENTS)
    B = load_and_upgrade("/workspace/circle.json", TARGET_SEGMENTS)

    print("⚙️ Generating morph sequence...")
    morph_sequence = generate_morph_sequence(A, B, TARGET_SEGMENTS, NUM_FRAMES)
    print(f"✅ Output shape: {morph_sequence.shape}")

    # 💾 Save JSON for curve engine
    with open(JSON_PATH, 'w') as f:
        json.dump({
            "points": morph_sequence.tolist(),
            "meta": {
                "target_segments": TARGET_SEGMENTS,
                "num_frames": NUM_FRAMES,
                "format": "[Control, Anchor, Control, Anchor, ...] per frame"
            }
        }, f, indent=2)
    print(f"💾 Saved JSON to {JSON_PATH}")

    # 🎬 Setup Figure for GIF
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_aspect('equal')
    ax.axis('off')

    # Auto-fit bounds with padding
    all_pts = morph_sequence.reshape(-1, 2)
    pad = 0.4
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    curve_line, = ax.plot([], [], 'b-', lw=2.2)
    anchor_dots, = ax.plot([], [], 'ro', ms=4, alpha=0.7)
    title_txt = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                        ha='center', fontsize=11)

    def update(f):
        pts = morph_sequence[f]  # [C, A, C, A, ...]
        curve = evaluate_bezier_curve(pts, samples_per_seg=64)

        curve_line.set_data(curve[:, 0], curve[:, 1])
        anchor_dots.set_data(pts[1::2, 0], pts[1::2, 1])  # Only plot anchors
        title_txt.set_text(
            f"Frame {f}/{NUM_FRAMES-1} | α = {f/(NUM_FRAMES-1):.2f}")
        return curve_line, anchor_dots, title_txt

    print("🎬 Rendering GIF (requires `pip install pillow`)...")
    FuncAnimation(fig, update, frames=NUM_FRAMES, blit=False, interval=33).save(
        GIF_PATH, fps=30, writer='pillow')
    print(f"✅ Saved GIF to {GIF_PATH}")

    plt.close(fig)  # Prevents hanging plot window
