from pathlib import Path
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox
import sys
import numpy as np

# Ensure spatial_pipeline is importable
_GUI_DIR = Path(__file__).resolve().parent
_SRC_DIR = _GUI_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from spatial_pipeline.config import PROJECT_ROOT, BSROFORMER_ROOT

BG = "#C8A97E"        
BG_2 = "#2E241C"      
PANEL_BG = "#433126"  
PANEL_BG2 = "#5E7D4B" 

ACCENT = "#D6B347"    
ACCENT2 = "#7FA36B"   
BORDER = "#6F5642"
TEXT = "#F7F1E8"
TEXT_DIM = "#D2C3AF"

CANVAS_BG = "#241B15"
GRID_COL = "#5A4A36"

FONT_APP_TITLE = ("Helvetica", 20, "bold")
FONT_SECTION = ("Helvetica", 13, "bold")
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
    # Recorded movement gesture: list of (t_seconds, azimuth_deg, elevation_deg)
    # sampled while the "Record Movement" button was held down. None means
    # "use the static azimuth/elevation instead" (default behaviour).
    recorded_movement: list[tuple[float, float, float]] | None = None


class AppState:
    def __init__(self):
        initial_sources = [
        ("vocals", "#D9B44A", 0),   # mustard gold
        ("drums", "#E27C3B", 35),   # terracotta
        ("bass", "#A65E2E", -35),   # burnt wood
        ("guitar", "#6B8E23", -23), # olive green
        ("piano", "#8FBC8F", 23),   # sage green
        ("other", "#2F7F73", -11),  # deep teal
    ]

        self.sources = [
            SourceState(name, color, azimuth=az)
            for name, color, az in initial_sources
        ]
        self.song_path = None
        self.demix_model_path = str(
        BSROFORMER_ROOT
        / "models"
        / "roformer-model-bs-roformer-sw-by-jarredou"
        / "BS-Rofo-SW-Fixed.ckpt"
        )
        self.renderer = "binaural"
        self.layout_path = None
        self.hrtf_path = None
        self.out_dir = PROJECT_ROOT / "outputs"
        self.hoa_order = 3
        self.output_name = ""
        self.last_output_path: str | None = None


def populate_sources_from_stem_paths(state: AppState, stems: dict[str, str]):
    for src in state.sources:
        stem_file = stems.get(src.name)
        if stem_file:
            src.wav_path = stem_file


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
    from spatial_pipeline.demix import demix_track_single

    if not state.song_path:
        raise ValueError("No song file selected. Browse a song first.")
    if not state.demix_model_path:
        raise ValueError("No model checkpoint selected. Load a .ckpt file first.")

    song_path = Path(state.song_path)
    out_dir = state.out_dir / "demixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    status.set(f"Demixing '{song_path.name}'… (this may take a while)")

    def _progress(msg):
        status.after(0, lambda: status.set(msg))

    all_results = demix_track_single(
        audio_path=str(song_path),
        output_dir=str(out_dir),
        model_path=state.demix_model_path,
        progress_callback=_progress,
    )

    song_key = song_path.stem
    if song_key not in all_results:
        raise RuntimeError(
            f"Demix finished but '{song_key}' not found in results. "
            f"Got: {list(all_results.keys())}"
        )

    stems = all_results[song_key]
    populate_sources_from_stem_paths(state, stems)

    status.set(f"Demix complete — stems in {out_dir}")

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
        btn.config(state="normal", text="▶ GENERATE")


def _do_generate(state: AppState, status):
    from spatial_pipeline.pipeline import (
        encode_stems_to_hoa,
        render_binaural_scene,
        decode_scene_for_layout,
    )
    from spatial_pipeline.config import DEFAULT_HRTF_SOFA, MEASUREMENTS_CSV
    from spatial_pipeline.frame_processing import split_frames
    from spatial_pipeline.ambisonics.core.trajectories import (
        generate_from_recording,
        generate_static,
    )
    import soundfile as sf

    # Frame parameters must match those used internally by encode_stems_to_hoa,
    # so that a recorded trajectory has exactly one position per frame.
    frame_size = 1024
    hop_size = 512

    stem_paths = {}
    positions = {}
    trajectories = {}
    any_recorded = any(
        src.recorded_movement and not src.mute for src in state.sources
    )

    for src in state.sources:
        if src.mute:
            continue
        if src.wav_path is None:
            raise ValueError(
                f"No WAV file selected for stem '{src.name}'. "
                "Load a WAV or mute unused stems."
            )

        stem_paths[src.name] = src.wav_path
        positions[src.name] = (-src.azimuth, src.elevation)

        if not any_recorded:
            continue

        info = sf.info(src.wav_path)
        n_frames = len(split_frames(np.zeros(info.frames), frame_size, hop_size))

        if src.recorded_movement:
            frame_hop_seconds = hop_size / info.samplerate

            # The recording is captured in screen/azimuth convention; positions_deg
            # elsewhere negates azimuth before encoding (see -src.azimuth above), so
            # we apply the same negation here for consistency between static and
            # recorded sources.
            negated = [(t, -azi, ele) for t, azi, ele in src.recorded_movement]

            trajectories[src.name] = generate_from_recording(
                n_frames=n_frames,
                recording=negated,
                frame_hop_seconds=frame_hop_seconds,
            )
        else:
            # encode_stems_to_hoa requires a trajectory for EVERY active stem
            # once the trajectories dict is non-None, so stems without a
            # recorded movement fall back to their static anchor position.
            trajectories[src.name] = generate_static(n_frames, -src.azimuth, src.elevation)

    if not stem_paths:
        raise ValueError("All stems are muted — nothing to render.")

    out_dir = state.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hoa_dir = out_dir / f"hoa{state.hoa_order}"
    rendered_dir = out_dir / "rendered"
    hoa_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    custom_output_name = state.output_name.strip()
    if custom_output_name:
        song_name = custom_output_name
    else:
        song_name = Path(state.song_path).stem if state.song_path else "output"

    hoa_path = str(hoa_dir / f"{song_name}_scene_hoa{state.hoa_order}.wav")

    status.set("Encoding stems to HOA…")
    encode_stems_to_hoa(
        stem_paths=stem_paths,
        positions_deg=positions,
        out_path=hoa_path,
        order=state.hoa_order,
        gains_db={src.name: src.gain_db for src in state.sources},
        trajectories=trajectories or None,
    )

    hrtf = str(state.hrtf_path) if state.hrtf_path else str(DEFAULT_HRTF_SOFA)
    layout = str(state.layout_path) if state.layout_path else str(MEASUREMENTS_CSV)
    renderer = state.renderer

    if renderer == "binaural":
        status.set("Rendering binaural…")
        # Always enforce the "_binaural" suffix on the output filename so that
        # evaluate_spatial.py can reliably recognize this as a binaural render,
        # regardless of the (optional) custom name typed by the user.
        binaural_name = song_name if song_name.lower().endswith("_binaural") else f"{song_name}_binaural"
        out = str(rendered_dir / f"{binaural_name}.wav")
        render_binaural_scene(hoa_path, hrtf, out, order=state.hoa_order)

    elif renderer == "layout_speaker":
        layout_csv = str(state.layout_path) if state.layout_path else str(MEASUREMENTS_CSV)
        layout_name = Path(layout_csv).stem
        status.set(f"Decoding to speaker layout '{layout_name}'…")
        out = str(rendered_dir / f"{song_name}_{layout_name}.wav")
        decode_scene_for_layout(hoa_path, out, layout_csv=layout_csv, order=state.hoa_order)

    else:
        raise ValueError(f"Unknown renderer: {renderer}")

    state.last_output_path = out
    status.set(f"Done! Output saved to {out}")
    messagebox.showinfo("Done", f"Output saved:\n{out}")