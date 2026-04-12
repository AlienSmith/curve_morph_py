import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .pipeline import generate_morph


def make_shape(cx, cy, r1, r2, n=128, phase=0.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
    return np.stack([cx + r1*np.cos(t), cy + r2*np.sin(t)], axis=1)


def run_morph_test():
    print("🔹 Generating test shapes...")
    A = make_shape(0.0, 0.0, 1.0, 0.6)  # Ellipse
    B = make_shape(0.2, 0.1, 0.8, 0.8, phase=np.pi/4)  # Rotated circle

    print("🔹 Running PBD pipeline...")
    frames, boundary_idx = generate_morph(A, B, num_frames=40)

    print("🔹 Rendering GIF...")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    sc_boundary = ax.scatter([], [], s=10, c='steelblue', zorder=3)
    sc_interior = ax.scatter([], [], s=4, c='lightgray', alpha=0.4, zorder=2)
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=11)

    def update(f):
        all_pts = frames[f]
        sc_boundary.set_offsets(all_pts[boundary_idx])
        sc_interior.set_offsets(np.delete(all_pts, boundary_idx, axis=0))
        title.set_text(f"Frame {f} | {len(boundary_idx)} boundary pts")
        return sc_boundary, sc_interior, title

    FuncAnimation(fig, update, frames=len(frames), blit=False, interval=30).save(
        "pbd_morph_test.gif", fps=24, writer='pillow')
    print("✅ Saved: pbd_morph_test.gif")


if __name__ == "__main__":
    run_morph_test()
