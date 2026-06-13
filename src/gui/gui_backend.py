from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox
import sys

# Ensure spatial_pipeline is importable
_GUI_DIR = Path(__file__).resolve().parent
_SRC_DIR = _GUI_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from spatial_pipeline.config import DEFAULT_GUI_DIR

BG = "#070a12"
BG_2 = "#0b1220"
PANEL_BG = "#122038"
PANEL_BG2 = "#1a2b46"

ACCENT = "#627df7"
ACCENT2 = "#279db4"
BORDER = "#31476f"

TEXT = "#edf2ff"
TEXT_DIM = "#8ea0c2"

CANVAS_BG = "#070a12"
GRID_COL = "#1f3354"

FONT_APP_TITLE = ("Helvetica", 15, "bold")
FONT_SECTION = ("Helvetica", 11, "bold")
FONT_LABEL = ("Helvetica", 10)
FONT_SMALL = ("Helvetica", 9)
FONT_MONO = ("Courier", 10)


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
        initial_sources = [
            ("vocals", "#a278ff", 0),   
            ("drums",  "#eef093", 35),  
            ("bass",   "#e8a146", -35), 
            ("guitar", "#6f8cff", -23), 
            ("piano",  "#d88ecf", 23),  
            ("other",  "#39a9c3", -11), 
        ]

        self.sources = [
            SourceState(name, color, azimuth=az)
            for name, color, az in initial_sources
        ]
        self.song_path = None
        self.demix_model_path = None
        self.renderer = "binaural"
        self.layout_path = None
        self.hrtf_path = None
        self.out_dir = DEFAULT_GUI_DIR
        self.hoa_order = 3


def run_demix_and_populate(state, status, btn: tk.Button, on_done_callback):
    """Background thread: demix the song, populate SourceState paths and positions."""
    btn.config(state="disabled", text="⏳ Demixing…")
    try:
        _do_demix(state, status, on_done_callback)
    except Exception as e:
        status.set(f"DEMIX ERROR: {e}")
        messagebox.showerror("Demix failed", str(e))
    finally:
        btn.config(state="normal", text="🎵 Demix Song")


def _do_demix(state, status, on_done_callback):
    from spatial_pipeline.demix import demix_folder
    from spatial_pipeline.scene_defaults import DEFAULT_POSITIONS_DEG

    if not state.song_path:
        raise ValueError("No song file selected. Browse a song first.")
    if not state.demix_model_path:
        raise ValueError("No model checkpoint selected. Load a .ckpt file first.")

    song_path = Path(state.song_path)
    out_dir = state.out_dir / "demixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    status.set(f"Demixing '{song_path.name}'… (this may take a while)")

    all_results = demix_folder(
        input_dir=str(song_path.parent),
        output_dir=str(out_dir),
        model_path=state.demix_model_path,
    )

    song_key = song_path.stem
    if song_key not in all_results:
        raise RuntimeError(
            f"Demix finished but '{song_key}' not found in results. "
            f"Got: {list(all_results.keys())}"
        )

    stems = all_results[song_key]   # { "vocals": "/path/vocals.wav", ... }

    for src in state.sources:
        stem_file = stems.get(src.name)
        if stem_file:
            src.wav_path = stem_file
        if src.name in DEFAULT_POSITIONS_DEG:
            src.azimuth, src.elevation = DEFAULT_POSITIONS_DEG[src.name]

    status.set(f"Demix complete — stems in {out_dir}")

    # Schedule GUI refresh on the main thread (Tkinter is not thread-safe)
    if on_done_callback:
        status.after(0, on_done_callback)


def run_generate(state, status, btn: tk.Button):
    btn.config(state="disabled", text="⏳ Generating…")
    try:
        _do_generate(state, status)
    except Exception as e:
        status.set(f"ERROR: {e}")
        messagebox.showerror("Generation failed", str(e))
    finally:
        btn.config(state="normal", text="▶  GENERATE")


def _do_generate(state: AppState, status):
    from pathlib import Path
    from spatial_pipeline.pipeline import (
        encode_stems_to_hoa,
        render_binaural_scene,
        render_ls17_binaural_scene,
        decode_scene_for_ls17,
    )
    from spatial_pipeline.config import DEFAULT_HRTF_SOFA, MEASUREMENTS_CSV

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

    song_name = Path(state.song_path).stem if state.song_path else "output"

    hoa_path = str(out_dir / f"{song_name}_scene_hoa{state.hoa_order}.wav")

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
        out = str(out_dir / f"{song_name}_binaural.wav")
        render_binaural_scene(hoa_path, hrtf, out, order=state.hoa_order)

    elif renderer == "ls17_binaural":
        status.set("Rendering LS17 → binaural…")
        out = str(out_dir / f"{song_name}_ls17_binaural.wav")
        render_ls17_binaural_scene(hoa_path, hrtf, out, order=state.hoa_order)

    elif renderer == "ls17":
        status.set("Decoding to LS17…")
        out = str(out_dir / f"{song_name}_17ch.wav")
        decode_scene_for_ls17(hoa_path, out, order=state.hoa_order)

    else:
        raise ValueError(f"Unknown renderer: {renderer}")

    status.set(f"Done! Output saved to {out}")
    messagebox.showinfo("Done", f"Output saved:\n{out}")