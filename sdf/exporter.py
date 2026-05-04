import struct
import json
import numpy as np
import io
from typing import Optional

MORPH_MAGIC = b"MORP"  # 4-byte magic identifier


def export_morph_to_buffer(sequence: np.ndarray, t_vals: np.ndarray, fps_hint: float = 20.0) -> io.BytesIO:
    """Export morph to an in-memory bytes buffer instead of writing to disk"""
    num_frames, num_points, _ = sequence.shape
    min_x, min_y = sequence.min(axis=(0, 1))
    max_x, max_y = sequence.max(axis=(0, 1))

    header = {
        "version": 1,
        "num_frames": num_frames,
        "num_points": num_points,
        "fps_hint": fps_hint,
        "bounds": {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y)
        },
        "user_metadata": {}
    }

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    buf = io.BytesIO()
    buf.write(MORPH_MAGIC)
    buf.write(struct.pack("<I", header_len))
    buf.write(header_json)
    buf.write(t_vals.tobytes(order="C"))
    buf.write(sequence.tobytes(order="C"))
    buf.seek(0)
    return buf
