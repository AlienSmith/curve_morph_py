from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form  # Add Form to your imports
import os
import json
import uuid
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from loader import upgrade_to_uniform_resolution
from utils import evaluate_bezier_curve
from morph import generate_morph_sequence

TARGET_SEGMENTS = 64
NUM_FRAMES = 61
# Force headless mode for the server
matplotlib.use('Agg')

app = FastAPI()

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp_renders")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- HELPER LOGIC ---


def process_shape_data(content, target_segments=64):
    """
    Extracts points from JSON (raw or manifest), upgrades resolution,
    and enforces CCW winding for FFT stability.
    """
    # Extract points from Manifest {"meta":..., "points":...} or Raw Array [[x,y],...]
    if isinstance(content, dict) and "points" in content:
        editor_pts = np.array(content["points"], dtype=np.float32)
    else:
        editor_pts = np.array(content, dtype=np.float32)

    if editor_pts.shape != (32, 2):
        raise ValueError(f"Expected (32, 2) array, got {editor_pts.shape}")

    # 1. Upgrade resolution to uniform arc-length anchors
    pts = upgrade_to_uniform_resolution(editor_pts, target_segments)

    # 2. Enforce CCW Winding (Shoelace formula)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]
                         ) + (x[-1] * y[0] - x[0] * y[-1]))

    if area < 0:
        # Flip to CCW and roll by 1 to keep [Anchor, Control] alignment
        pts = pts[::-1]
        pts = np.roll(pts, 1, axis=0)
        print("🔄 Fixed Winding: Flipped CW to CCW for FFT.")

    return pts

# --- API ENDPOINTS ---


@app.post("/api/generate-morph")
async def generate_morph(
    background_tasks: BackgroundTasks,
    fileA: UploadFile = File(...),
    fileB: UploadFile = File(...),
    dpi: int = Form(100)  # Receive DPI as a form parameter
):
    job_id = str(uuid.uuid4())
    gif_path = os.path.join(TEMP_DIR, f"{job_id}.gif")

    try:
        json_a = json.loads(await fileA.read())
        json_b = json.loads(await fileB.read())
        ptsA = process_shape_data(json_a, TARGET_SEGMENTS)
        ptsB = process_shape_data(json_b, TARGET_SEGMENTS)
        print("⚙️ Generating morph sequence...")
        morph_sequence = generate_morph_sequence(
            ptsA, ptsB, TARGET_SEGMENTS, num_frames=NUM_FRAMES)
        print(f"✅ Output shape: {morph_sequence.shape}")
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

    # Dynamic scaling: thicker lines for higher DPI
    line_weight = 2 * (dpi / 100)
    marker_size = 3 * (dpi / 100)

    # 🎬 Setup Figure for GIF
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_aspect('equal')
    ax.axis('off')

    # Auto-fit bounds with padding
    all_pts = morph_sequence.reshape(-1, 2)
    pad = 0.4
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    curve_line, = ax.plot([], [], 'b-', lw=line_weight)
    anchor_dots, = ax.plot([], [], 'ro', ms=marker_size, alpha=0.7)
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
    anim = FuncAnimation(fig, update, frames=41, blit=True)

    # Using 'optimize=True' can help with file size but takes a bit more CPU
    anim.save(gif_path, writer='pillow', fps=20)
    plt.close(fig)

    background_tasks.add_task(os.remove, gif_path)
    return FileResponse(gif_path, media_type="image/gif")

app.add_middleware(
    CORSMiddleware,
    # Allows all domains (including 5173, 5174, etc.)
    allow_origins=["*"],
    allow_credentials=True,          # Keep this True for session/cookie support
    allow_methods=["*"],             # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],             # Allows all custom headers
)
# Mount static files for the Editor and Previewer
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    print(f"🚀 Server started. Editor at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
