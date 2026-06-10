from pathlib import Path
from collections import defaultdict
import sys

# When the code is run with flag "--trajectories", it will generate trajectory test files (orbit, flyover, bounce) in addition to static positions.
# When the code is run with flag "--static-only", it will generate only static position test files (default behavior).


# Decoder generation mode. Valid values:
#   "external_only" - generate HOA scene files for an external decoder only
#   "internal_only" - generate decoded output using the internal LS17 decoder
#   "both"          - generate both HOA scene files and internal decoded output
GENERATE_FOR_DECODER = "internal_only"
VALID_DECODER_GENERATION_MODES = {"external_only", "internal_only", "both"}
if GENERATE_FOR_DECODER not in VALID_DECODER_GENERATION_MODES:
    raise ValueError(
        f"Unsupported GENERATE_FOR_DECODER value: {GENERATE_FOR_DECODER}. "
        f"Choose one of {sorted(VALID_DECODER_GENERATION_MODES)}."
    )

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import encode_stems_to_hoa, decode_scene_for_ls17
from spatial_pipeline.scene_defaults import STEM_TYPES
from spatial_pipeline.audio_io import load_mono
from spatial_pipeline.frame_processing import split_frames
from spatial_pipeline.ambisonics.core.trajectories import (
    generate_static,
    generate_orbit,
    generate_arc_flyover,
    generate_bounce
)
import argparse

"""
generate_decoder_test_files.py
==============================

Utility script used exclusively for HOA decoder validation and trajectory testing.

This script generates a set of synthetic HOA test scenes where
all stems are collapsed into a single spatial position or trajectory.

The purpose is to verify that:

    Audio Stem
        ↓
    HOA Encoder
        ↓
    HOA File (3rd Order / SN3D)
        ↓
    AllRAD Decoder
        ↓
    Loudspeaker Layout
        ↓
    Perceived Source Position

is working correctly.

Generated test files:

    Static positions:
    *_all_front_hoa3.wav
    *_all_left_hoa3.wav
    *_all_right_hoa3.wav
    *_all_back_hoa3.wav

    Trajectories (optional):
    *_orbit_hoa3.wav
    *_flyover_hoa3.wav
    *_bounce_hoa3.wav

Expected behaviour:

    all_front  -> source perceived in front
    all_left   -> source perceived on the left
    all_right  -> source perceived on the right
    all_back   -> source perceived behind the listener
    orbit      -> source circles around the listener
    flyover    -> source flies over the listener's head
    bounce     -> source bounces left and right

These files are intended only for system validation and
decoder debugging.

They are NOT part of the normal content production pipeline.

This script can be removed once a proper GUI-based testing
workflow is implemented.
"""

# TEST POSITIONS (static)
TEST_POSITIONS = {
    "all_front": (0.0, 0.0),
    "all_left": (90.0, 0.0),
    "all_right": (-90.0, 0.0),
    "all_back": (180.0, 0.0),
}

# TEST TRAJECTORIES (dynamic)
TEST_TRAJECTORIES = {
    "orbit": (generate_orbit, {"start_azi_deg": 0.0, "rotations": 1.0, "elevation_deg": 0.0}),
    "flyover": (generate_arc_flyover, {"azimuth_deg": 0.0, "start_ele_deg": -90.0, "end_ele_deg": 90.0}),
    "bounce": (generate_bounce, {"center_azi_deg": 0.0, "width_deg": 45.0, "bounces": 10.0, "elevation_deg": 0.0}),
}

def collect_stems_by_song(output_folder: Path):
    all_songs_stems = defaultdict(dict)

    for wav_path in output_folder.glob("*.wav"):

        if wav_path.stem.endswith("_3d_scene"):
            continue

        if wav_path.stem.endswith("_hoa3"):
            continue

        for stem in STEM_TYPES:
            suffix = f"_{stem}"

            if wav_path.stem.endswith(suffix):
                song_name = wav_path.stem[:-len(suffix)]
                all_songs_stems[song_name][stem] = str(wav_path)
                break

    return dict(all_songs_stems)


def main():

    parser = argparse.ArgumentParser(
        description="Generate HOA test files for decoder validation and trajectory testing"
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

    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    print("\n=== HOA DECODER TEST FILE GENERATOR ===\n")
    print(f"Generation mode: {GENERATE_FOR_DECODER}\n")

    all_songs_stems = collect_stems_by_song(output_folder)

    if not all_songs_stems:
        print("No stems found.")
        return

    # Decide what to generate
    generate_static_tests = True
    generate_trajectory_tests = args.trajectories
    generate_internal_decode = GENERATE_FOR_DECODER in ("internal_only", "both")

    for song_name, stem_paths in all_songs_stems.items():

        missing_stems = [
            stem
            for stem in STEM_TYPES
            if stem not in stem_paths
        ]

        if missing_stems:
            print(
                f"Skipping '{song_name}' "
                f"(missing stems {missing_stems})"
            )
            continue

        # Generate static position tests
        if generate_static_tests:
            print(f"\n--- Static Position Tests for '{song_name}' ---")
            for test_name, (azi_deg, ele_deg) in TEST_POSITIONS.items():

                print(
                    f"Generating {test_name} "
                    f"for '{song_name}'..."
                )

                out_path = (
                    output_folder
                    / f"{song_name}_{test_name}_hoa3.wav"
                )

                encode_stems_to_hoa(
                    stem_paths=stem_paths,
                    positions_deg={stem: (azi_deg, ele_deg) for stem in STEM_TYPES},
                    out_path=str(out_path),
                    order=3,
                    normalization="sn3d",
                    trajectory_fn=generate_static,
                )

                print(f"Saved: {out_path.name}")

                if generate_internal_decode:
                    decoded_path = output_folder / f"{song_name}_{test_name}_17ch.wav"
                    decode_scene_for_ls17(
                        scene_path=str(out_path),
                        out_path=str(decoded_path),
                        order=3,
                    )
                    print(f"Decoded: {decoded_path.name}")

                    if GENERATE_FOR_DECODER == "internal_only":
                        out_path.unlink(missing_ok=True)
                        print(f"Removed HOA scene file: {out_path.name}")

        # Generate trajectory tests
        if generate_trajectory_tests:
            print(f"\n--- Trajectory Tests for '{song_name}' ---")
            
            # Estimate n_frames once for all trajectories
            frame_size = 1024
            hop_size = 512
            first_stem_path = list(stem_paths.values())[0]
            signal, _ = load_mono(first_stem_path)
            n_frames = len(split_frames(signal, frame_size, hop_size))
            
            for test_name, (trajectory_fn, kwargs) in TEST_TRAJECTORIES.items():

                try:
                    print(
                        f"Generating {test_name} trajectory "
                        f"for '{song_name}'..."
                    )

                    out_path = (
                        output_folder
                        / f"{song_name}_{test_name}_hoa3.wav"
                    )

                    # Generate trajectories for each stem
                    trajectories = {
                        stem: trajectory_fn(n_frames, **kwargs)
                        for stem in STEM_TYPES
                    }

                    # Extract initial azimuth for positions_deg (used as fallback, not primary when trajectories provided)
                    initial_azi = next(iter(kwargs.values())) if kwargs else 0.0

                    encode_stems_to_hoa(
                        stem_paths=stem_paths,
                        positions_deg={stem: (initial_azi, 0.0) for stem in STEM_TYPES},
                        out_path=str(out_path),
                        order=3,
                        normalization="sn3d",
                        trajectory_fn=trajectory_fn,
                        trajectories=trajectories,
                    )

                    print(f"Saved: {out_path.name}")

                    if generate_internal_decode:
                        decoded_path = output_folder / f"{song_name}_{test_name}_17ch.wav"
                        decode_scene_for_ls17(
                            scene_path=str(out_path),
                            out_path=str(decoded_path),
                            order=3,
                        )
                        print(f"Decoded: {decoded_path.name}")

                        if GENERATE_FOR_DECODER == "internal_only":
                            out_path.unlink(missing_ok=True)
                            print(f"Removed HOA scene file: {out_path.name}")
                    
                except Exception as e:
                    print(f"ERROR generating {test_name}: {e}")
                    import traceback
                    traceback.print_exc()

    print("\nDone.\n")


if __name__ == "__main__":
    main()