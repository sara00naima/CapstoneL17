from pathlib import Path
from collections import defaultdict
import sys

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import encode_stems_to_hoa
from spatial_pipeline.scene_defaults import STEM_TYPES
from spatial_pipeline.audio_io import load_mono
from spatial_pipeline.frame_processing import split_frames
from spatial_pipeline.ambisonics.core.trajectories import (
    generate_static,
    generate_orbit,
    generate_arc_flyover,
    generate_bounce
)

"""
generate_decoder_test_files.py
==============================

Utility script used exclusively for HOA decoder validation.

This script generates a set of synthetic HOA test scenes where
all stems are collapsed into a single static spatial position.

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

    *_all_front_hoa3.wav
    *_all_left_hoa3.wav
    *_all_right_hoa3.wav
    *_all_back_hoa3.wav

Expected behaviour:

    all_front  -> source perceived in front
    all_left   -> source perceived on the left
    all_right  -> source perceived on the right
    all_back   -> source perceived behind the listener

These files are intended only for system validation and
decoder debugging.

They are NOT part of the normal content production pipeline.

This script can be removed once a proper GUI-based testing
workflow is implemented.
"""

# TEST POSITIONS
TEST_LAYOUTS = {
    "all_front": (0.0, 0.0),
    "all_left": (90.0, 0.0),
    "all_right": (-90.0, 0.0),
    "all_back": (180.0, 0.0),
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

    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    print("\n=== HOA DECODER TEST FILE GENERATOR ===\n")

    all_songs_stems = collect_stems_by_song(output_folder)

    if not all_songs_stems:
        print("No stems found.")
        return

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

    
        for test_name, (azi_deg, ele_deg) in TEST_LAYOUTS.items():

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

    print("\nDone.\n")


if __name__ == "__main__":
    main()