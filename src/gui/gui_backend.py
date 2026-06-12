from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox


BG = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
TEXT = "#eaeaea"
TEXT_DIM = "#7f8c9b"
CANVAS_BG = "#0d1117"
GRID_COL = "#1e2d40"
SOURCE_COLS = ["#4fc3f7", "#81c784", "#ffb74d", "#ba68c8", "#f06292", "#4dd0e1"]
STEM_TYPES = ["vocals", "drums", "bass", "guitar", "piano", "other"]


@dataclass
class SourceState:
    name: str
    color: str
    azimuth: float = 0.0
    elevation: float = 0.0
    gain_db: float = 0.0
    mute: bool = False
    solo: bool = False
    wav_path: str | None = None


class AppState:
    def __init__(self):
        self.sources = [SourceState(n, c) for n, c in zip(STEM_TYPES, SOURCE_COLS)]
        self.song_path = None
        self.renderer = "binaural"
        self.layout_path = None
        self.hrtf_path = None
        self.out_dir = Path("outputs/gui")
        self.hoa_order = 3


def run_generate(state: AppState, status, btn: tk.Button):
    btn.config(state="disabled", text="⏳ Generating…")
    try:
        _do_generate(state, status)
    except Exception as e:
        status.set(f"ERROR: {e}")
        messagebox.showerror("Generation failed", str(e))
    finally:
        btn.config(state="normal", text="▶  GENERATE")


def _do_generate(state: AppState, status):
    try:
        import sys
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from spatial_pipeline.pipeline import (
            encode_stems_to_hoa,
            render_binaural_scene,
            render_ls17_binaural_scene,
            decode_scene_for_ls17,
        )
        from spatial_pipeline.config import DEFAULT_HRTF_SOFA, MEASUREMENTS_CSV
    except ImportError as e:
        raise RuntimeError(
            f"Could not import spatial_pipeline: {e}\n"
            "Run this script from the project root with src/ on the path."
        )

    stem_paths = {}
    positions = {}

    for src in state.sources:
        if src.mute:
            continue
        if src.wav_path is None:
            raise ValueError(
                f"No WAV file selected for stem '{src.name}'. "
                "Load a WAV or mute unused stems."
            )
        stem_paths[src.name] = src.wav_path
        positions[src.name] = (src.azimuth, src.elevation)

    if not stem_paths:
        raise ValueError("All stems are muted — nothing to render.")

    out_dir = state.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hoa_path = str(out_dir / f"scene_hoa{state.hoa_order}.wav")

    status.set("Encoding stems to HOA…")
    encode_stems_to_hoa(
        stem_paths=stem_paths,
        positions_deg=positions,
        out_path=hoa_path,
        order=state.hoa_order,
    )

    hrtf = str(state.hrtf_path) if state.hrtf_path else str(DEFAULT_HRTF_SOFA)
    layout = str(state.layout_path) if state.layout_path else str(MEASUREMENTS_CSV)
    renderer = state.renderer

    if renderer == "binaural":
        status.set("Rendering binaural…")
        out = str(out_dir / "output_binaural.wav")
        render_binaural_scene(hoa_path, hrtf, out, order=state.hoa_order)

    elif renderer == "ls17_binaural":
        status.set("Rendering LS17 → binaural…")
        out = str(out_dir / "output_ls17_binaural.wav")
        render_ls17_binaural_scene(hoa_path, hrtf, out, order=state.hoa_order)

    elif renderer == "ls17":
        status.set("Decoding to LS17…")
        out = str(out_dir / "output_17ch.wav")
        decode_scene_for_ls17(hoa_path, out, order=state.hoa_order)

    else:
        raise ValueError(f"Unknown renderer: {renderer}")

    status.set(f"Done! Output saved to {out}")
    messagebox.showinfo("Done", f"Output saved:\n{out}")