from pathlib import Path
import sys
import argparse


# When the code is run with flag "--trajectories", it will generate trajectory test files (orbit, flyover, bounce) in addition to static positions.
# When the code is run with flag "--static-only", it will generate only static position test files (default behavior).


# Decoder generation mode. Valid values:
#   "external_only" - generate HOA scene files for an external decoder only
#   "internal_only" - generate decoded output using the internal LS17 decoder
#   "both"          - generate both HOA scene files and internal decoded output
GENERATE_FOR_DECODER = "both"
VALID_DECODER_GENERATION_MODES = {"external_only", "internal_only", "both"}
if GENERATE_FOR_DECODER not in VALID_DECODER_GENERATION_MODES:
    raise ValueError(
        f"Unsupported GENERATE_FOR_DECODER value: {GENERATE_FOR_DECODER}. "
        f"Choose one of {sorted(VALID_DECODER_GENERATION_MODES)}."
    )


# Set to True to render a direct binaural WAV (bypasses LS17 decoder — tests encoder only).
GENERATE_BINAURAL = False


# Set to True to render a binaural WAV that routes through the LS17 decoder first
# (HOA → 17 speaker feeds → HRTF per speaker → stereo).
# This lets you audition the museum decoder on headphones.
GENERATE_LS17_BINAURAL = True


CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import (
    encode_stems_to_hoa,
    decode_scene_for_ls17,
    render_binaural_scene,
    render_ls17_binaural_scene,
)
from spatial_pipeline.config import (
    DEFAULT_HRTF_SOFA,
    DEFAULT_DEMIX_DIR,
    DEFAULT_TEST_HOA_DIR,
    DEFAULT_TEST_LS17_DIR,
    DEFAULT_TEST_BINAURAL_DIR,
    DEFAULT_TEST_LS17_BINAURAL_DIR,
)
from spatial_pipeline.scene_defaults import STEM_TYPES
from spatial_pipeline.audio_io import load_mono
from spatial_pipeline.frame_processing import split_frames
from spatial_pipeline.ambisonics.core.trajectories import (
    generate_static,
    generate_orbit,
    generate_arc_flyover,
    generate_bounce,
)


"""
generate_decoder_test_files.py
==============================

Utility script used exclusively for HOA decoder validation and trajectory testing.

This version processes one song at a time from:
    outputs/demixed/{song_name}-stems/

Examples:
    python generate_decoder_test_files.py
    python generate_decoder_test_files.py --song my-song
    python generate_decoder_test_files.py --trajectories
    python generate_decoder_test_files.py --song my-song --trajectories
"""


TEST_POSITIONS = {
    "all_front": (0.0, 0.0),
    "all_left": (90.0, 0.0),
    "all_right": (-90.0, 0.0),
    "all_back": (180.0, 0.0),
}

TEST_TRAJECTORIES = {
    "orbit": (
        generate_orbit,
        {"start_azi_deg": 0.0, "rotations": 1.0, "elevation_deg": 0.0},
    ),
    "flyover": (
        generate_arc_flyover,
        {"azimuth_deg": 0.0, "start_ele_deg": -90.0, "end_ele_deg": 90.0},
    ),
    "bounce": (
        generate_bounce,
        {"center_azi_deg": 0.0, "width_deg": 45.0, "bounces": 10.0, "elevation_deg": 0.0},
    ),
}


def collect_single_song_stems(song_dir: Path):
    stem_paths = {}
    unmatched_files = []

    if not song_dir.exists():
        raise FileNotFoundError(f"Song stems folder not found: {song_dir}")

    if not song_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {song_dir}")

    for wav_path in song_dir.glob("*.wav"):
        name = wav_path.stem

        if name.endswith("_3d_scene"):
            continue

        if name.endswith("_hoa3"):
            continue

        matched = False
        for stem in STEM_TYPES:
            if name.startswith(f"{stem}-"):
                stem_paths[stem] = str(wav_path)
                matched = True
                break

        if not matched:
            unmatched_files.append(wav_path.name)

    if unmatched_files:
        print("\nWARNING: these files did not match any known stem type:")
        for name in unmatched_files:
            print(f"  - {name}")

    return stem_paths


def get_initial_azimuth(kwargs):
    if "start_azi_deg" in kwargs:
        return kwargs["start_azi_deg"]
    if "azimuth_deg" in kwargs:
        return kwargs["azimuth_deg"]
    if "center_azi_deg" in kwargs:
        return kwargs["center_azi_deg"]
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Generate HOA test files for decoder validation and trajectory testing"
    )
    parser.add_argument(
        "--song",
        default="test_audio",
        help='Song name without "-stems" suffix, default: test_audio'
    )
    parser.add_argument(
        "--trajectories",
        action="store_true",
        help="Generate trajectory test files (orbit, flyover, bounce) in addition to static positions"
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Generate only static position test files (default behavior)"
    )
    args = parser.parse_args()

    song_name = args.song
    song_folder = (DEFAULT_DEMIX_DIR / f"{song_name}-stems").resolve()

    print("\n=== HOA DECODER TEST FILE GENERATOR ===\n")
    print(f"Decoder mode:      {GENERATE_FOR_DECODER}")
    print(f"Binaural:          {GENERATE_BINAURAL}")
    print(f"LS17 binaural:     {GENERATE_LS17_BINAURAL}")
    print(f"Song name:         {song_name}")
    print(f"Song stems folder: {song_folder}")
    print(f"Output root:       {DEFAULT_TEST_HOA_DIR.parent}\n")

    stem_paths = collect_single_song_stems(song_folder)

    if not stem_paths:
        print("No stems found.")
        return

    print("Found stems:")
    for stem, path in sorted(stem_paths.items()):
        print(f"  - {stem}: {path}")

    missing_stems = [stem for stem in STEM_TYPES if stem not in stem_paths]
    if missing_stems:
        print(f"\nSkipping '{song_name}' (missing stems {missing_stems})")
        return

    generate_static_tests = True
    generate_trajectory_tests = args.trajectories and not args.static_only
    generate_internal_decode = GENERATE_FOR_DECODER in ("internal_only", "both")
    generate_hoa_file = (
        GENERATE_FOR_DECODER in ("external_only", "both")
        or GENERATE_BINAURAL
        or GENERATE_LS17_BINAURAL
    )

    if generate_hoa_file:
        DEFAULT_TEST_HOA_DIR.mkdir(parents=True, exist_ok=True)
    if generate_internal_decode:
        DEFAULT_TEST_LS17_DIR.mkdir(parents=True, exist_ok=True)
    if GENERATE_BINAURAL:
        DEFAULT_TEST_BINAURAL_DIR.mkdir(parents=True, exist_ok=True)
    if GENERATE_LS17_BINAURAL:
        DEFAULT_TEST_LS17_BINAURAL_DIR.mkdir(parents=True, exist_ok=True)

    if generate_static_tests:
        print(f"\n--- Static Position Tests for '{song_name}' ---")
        for test_name, (azi_deg, ele_deg) in TEST_POSITIONS.items():
            print(f"  {test_name}...")

            hoa_path = DEFAULT_TEST_HOA_DIR / f"{song_name}_{test_name}_hoa3.wav"

            encode_stems_to_hoa(
                stem_paths=stem_paths,
                positions_deg={stem: (azi_deg, ele_deg) for stem in STEM_TYPES},
                out_path=str(hoa_path),
                order=3,
                normalization="sn3d",
                trajectory_fn=generate_static,
            )
            print(f"    HOA:            {hoa_path.name}")

            if generate_internal_decode:
                ls17_path = DEFAULT_TEST_LS17_DIR / f"{song_name}_{test_name}_17ch.wav"
                decode_scene_for_ls17(
                    scene_path=str(hoa_path),
                    out_path=str(ls17_path),
                    order=3,
                )
                print(f"    LS17 decoded:   {ls17_path.name}")

            if GENERATE_BINAURAL:
                binaural_path = DEFAULT_TEST_BINAURAL_DIR / f"{song_name}_{test_name}_binaural.wav"
                render_binaural_scene(
                    scene_path=str(hoa_path),
                    sofa_path=str(DEFAULT_HRTF_SOFA),
                    out_path=str(binaural_path),
                    order=3,
                )
                print(f"    Binaural:       {binaural_path.name}")

            if GENERATE_LS17_BINAURAL:
                ls17_binaural_path = DEFAULT_TEST_LS17_BINAURAL_DIR / f"{song_name}_{test_name}_ls17_binaural.wav"
                render_ls17_binaural_scene(
                    scene_path=str(hoa_path),
                    sofa_path=str(DEFAULT_HRTF_SOFA),
                    out_path=str(ls17_binaural_path),
                    order=3,
                )
                print(f"    LS17 binaural:  {ls17_binaural_path.name}")

            if GENERATE_FOR_DECODER == "internal_only":
                hoa_path.unlink(missing_ok=True)
                print("    HOA removed.")

    print(f"\n--- Song Test for '{song_name}' ---")

    song_test_positions = {
        "vocals": (0.0, 0.0),
        "guitar": (-90.0, 0.0),
        "drums": (90.0, 0.0),
        "bass": (180.0, 0.0),
        "other": (0.0, 90.0),
        "piano": (0.0, -20.0),
    }

    hoa_path = DEFAULT_TEST_HOA_DIR / f"{song_name}_song_test_hoa3.wav"

    encode_stems_to_hoa(
        stem_paths=stem_paths,
        positions_deg=song_test_positions,
        out_path=str(hoa_path),
        order=3,
        normalization="sn3d",
        trajectory_fn=generate_static,
    )
    print(f"    HOA:            {hoa_path.name}")

    if generate_internal_decode:
        ls17_path = DEFAULT_TEST_LS17_DIR / f"{song_name}_song_test_17ch.wav"
        decode_scene_for_ls17(
            scene_path=str(hoa_path),
            out_path=str(ls17_path),
            order=3,
        )
        print(f"    LS17 decoded:   {ls17_path.name}")

    if GENERATE_LS17_BINAURAL:
        ls17_binaural_path = DEFAULT_TEST_LS17_BINAURAL_DIR / f"{song_name}_song_test_ls17_binaural.wav"
        render_ls17_binaural_scene(
            scene_path=str(hoa_path),
            sofa_path=str(DEFAULT_HRTF_SOFA),
            out_path=str(ls17_binaural_path),
            order=3,
        )
        print(f"    LS17 binaural:  {ls17_binaural_path.name}")

    if generate_trajectory_tests:
        print(f"\n--- Trajectory Tests for '{song_name}' ---")

        frame_size = 1024
        hop_size = 512
        first_stem_path = list(stem_paths.values())[0]
        signal, _ = load_mono(first_stem_path)
        n_frames = len(split_frames(signal, frame_size, hop_size))

        for test_name, (trajectory_fn, kwargs) in TEST_TRAJECTORIES.items():
            try:
                print(f"  {test_name}...")

                hoa_path = DEFAULT_TEST_HOA_DIR / f"{song_name}_{test_name}_hoa3.wav"

                trajectories = {
                    stem: trajectory_fn(n_frames, **kwargs)
                    for stem in STEM_TYPES
                }

                initial_azi = get_initial_azimuth(kwargs)

                encode_stems_to_hoa(
                    stem_paths=stem_paths,
                    positions_deg={stem: (initial_azi, 0.0) for stem in STEM_TYPES},
                    out_path=str(hoa_path),
                    order=3,
                    normalization="sn3d",
                    trajectory_fn=trajectory_fn,
                    trajectories=trajectories,
                )
                print(f"    HOA:            {hoa_path.name}")

                if generate_internal_decode:
                    ls17_path = DEFAULT_TEST_LS17_DIR / f"{song_name}_{test_name}_17ch.wav"
                    decode_scene_for_ls17(
                        scene_path=str(hoa_path),
                        out_path=str(ls17_path),
                        order=3,
                    )
                    print(f"    LS17 decoded:   {ls17_path.name}")

                if GENERATE_BINAURAL:
                    binaural_path = DEFAULT_TEST_BINAURAL_DIR / f"{song_name}_{test_name}_binaural.wav"
                    render_binaural_scene(
                        scene_path=str(hoa_path),
                        sofa_path=str(DEFAULT_HRTF_SOFA),
                        out_path=str(binaural_path),
                        order=3,
                    )
                    print(f"    Binaural:       {binaural_path.name}")

                if GENERATE_LS17_BINAURAL:
                    ls17_binaural_path = DEFAULT_TEST_LS17_BINAURAL_DIR / f"{song_name}_{test_name}_ls17_binaural.wav"
                    render_ls17_binaural_scene(
                        scene_path=str(hoa_path),
                        sofa_path=str(DEFAULT_HRTF_SOFA),
                        out_path=str(ls17_binaural_path),
                        order=3,
                    )
                    print(f"    LS17 binaural:  {ls17_binaural_path.name}")

                if GENERATE_FOR_DECODER == "internal_only":
                    hoa_path.unlink(missing_ok=True)
                    print("    HOA removed.")

            except Exception as e:
                print(f"  ERROR generating {test_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\nDone.\n")


if __name__ == "__main__":
    main()