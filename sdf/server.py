import io

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
from fastapi.responses import FileResponse, StreamingResponse

from loader import upgrade_to_uniform_resolution
from utils import evaluate_bezier_curve
from morph import generate_morph_sequence
from loader import process_shape_data
from exporter import export_morph_to_buffer


TARGET_SEGMENTS = 64
NUM_FRAMES = 60
# Force headless mode for the server
matplotlib.use('Agg')

app = FastAPI()

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp_renders")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- API ENDPOINTS ---


@app.post("/api/generate-morph")
async def generate_morph(
    background_tasks: BackgroundTasks,
    fileA: UploadFile = File(...),
    fileB: UploadFile = File(...),
    dpi: int = Form(100),
    include_morph: bool = Form(True)  # Optional toggle to include .morph file
):
    job_id = str(uuid.uuid4())
    gif_path = os.path.join(TEMP_DIR, f"{job_id}.gif")

    try:
        json_a = json.loads(await fileA.read())
        json_b = json.loads(await fileB.read())
        ptsA = process_shape_data(json_a, TARGET_SEGMENTS)
        ptsB = process_shape_data(json_b, TARGET_SEGMENTS)
        print("⚙️ Generating morph sequence...")
        # ⚠️ Note: your generate_morph_sequence returns (sequence, t_vals) right?
        # I added t_vals here since you need it for the morph export
        morph_sequence, t_vals = generate_morph_sequence(
            ptsA, ptsB, TARGET_SEGMENTS, verbose=True)
        print(f"✅ Output shape: {morph_sequence.shape}, frames: {len(t_vals)}")
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

    # Generate GIF as before
    line_weight = 2 * (dpi / 100)
    marker_size = 3 * (dpi / 100)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_aspect('equal')
    ax.axis('off')

    all_pts = morph_sequence.reshape(-1, 2)
    pad = 0.4
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    curve_line, = ax.plot([], [], 'b-', lw=line_weight)
    anchor_dots, = ax.plot([], [], 'ro', ms=marker_size, alpha=0.7)
    title_txt = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                        ha='center', fontsize=11)

    def update(f):
        pts = morph_sequence[f]
        curve = evaluate_bezier_curve(pts, samples_per_seg=64)
        curve_line.set_data(curve[:, 0], curve[:, 1])
        anchor_dots.set_data(pts[1::2, 0], pts[1::2, 1])
        title_txt.set_text(
            f"Frame {f}/{len(morph_sequence)-1} | α = {t_vals[f]:.2f}")
        return curve_line, anchor_dots, title_txt

    anim = FuncAnimation(fig, update, frames=len(morph_sequence),
                         blit=True, interval=500)
    anim.save(gif_path, writer='pillow', fps=20)
    plt.close(fig)

    # Clean up GIF after response is sent
    background_tasks.add_task(os.remove, gif_path)

    # If user only wants GIF, return it as before
    if not include_morph:
        return FileResponse(gif_path, media_type="image/gif")

    # --- Package both GIF and .morph into multipart response ---
    # Read GIF into memory
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()

    # Generate .morph file in memory
    morph_buf = export_morph_to_buffer(morph_sequence, t_vals, fps_hint=20)
    morph_bytes = morph_buf.read()

    # Build multipart boundary
    boundary = f"----MorphBoundary{uuid.uuid4().hex}"
    body = io.BytesIO()

    # Add GIF part
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        b"Content-Disposition: form-data; name=\"gif\"; filename=\"animation.gif\"\r\n")
    body.write(b"Content-Type: image/gif\r\n\r\n")
    body.write(gif_bytes)
    body.write(b"\r\n")

    # Add .morph part
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        b"Content-Disposition: form-data; name=\"morph\"; filename=\"animation.morph\"\r\n")
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(morph_bytes)
    body.write(b"\r\n")

    # Final boundary marker
    body.write(f"--{boundary}--\r\n".encode())
    body.seek(0)

    return StreamingResponse(
        body,
        media_type=f"multipart/form-data; boundary={boundary}",
        headers={
            "Content-Disposition": f"attachment; filename=\"{job_id}.morph\""
        }
    )


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
