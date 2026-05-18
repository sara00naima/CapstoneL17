import sys
from collections import defaultdict
from pathlib import Path

from spatial_pipeline.pipeline import encode_stems_to_hoa
from spatial_pipeline.scene_defaults import STEM_TYPES, DEFAULT_POSITIONS_DEG

# Resolve directory structure relative to this script's location
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Add src/ to the Python path so that spatial_pipeline can be imported
# as a package without needing a formal install 
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def collect_stems_by_song(output_folder: Path) -> dict[str, dict[str, str]]:
    """
    Scans the output folder for stem wav files produced by the demixing stage
    and groups them by song name.

    Expected filename format: {song_name}_{stem_type}.wav

    Returns: { "my_song": { "vocals": "/path/to/my_song_vocals.wav", ... } }
    """
    all_songs_stems = defaultdict(dict)

    for wav_path in output_folder.glob("*.wav"):
        # Skip any already-encoded scene files from a previous run
        if wav_path.stem.endswith("_3d_scene"):
            continue
        
        # Check which stem type this file corresponds to by matching the filename suffix
        for stem in STEM_TYPES:
            suffix = f"_{stem}"
            if wav_path.stem.endswith(suffix):
                # Strip the stem suffix to recover the original song name
                song_name = wav_path.stem[: -len(suffix)]
                all_songs_stems[song_name][stem] = str(wav_path)
                break # a file can only match one stem type, no need to check further

    return dict(all_songs_stems)


def main():
    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    print("--- THE POSITIONING & ENCODING STAGE (HOA) ---")

    # group all stems produced by the demixing stage
    all_songs_stems = collect_stems_by_song(output_folder)
    if not all_songs_stems:
        print(f"No stems found in {output_folder}")
        return

    print("Assigning 3D coordinates using 3rd-Order / 16 Channels...")
    print("\nStarting Ambisonic Encoding...")

    for song_name, stem_paths in all_songs_stems.items():
        # Require all stem types to be present
        missing_stems = [stem for stem in STEM_TYPES if stem not in stem_paths]
        if missing_stems:
            print(f"Warning: '{song_name}' is missing stems {missing_stems}. Skipping.")
            continue
        
        # Output path for the encoded FOA scene
        final_out_path = output_folder / f"{song_name}_3d_scene_hoa3.wav"
        print(f"Building HOA scene for: {song_name}")

        # Encode all stems into a single High Order Ambisonics bus
        encode_stems_to_hoa(
            stem_paths=stem_paths,
            positions_deg=DEFAULT_POSITIONS_DEG,
            out_path=str(final_out_path),
            order=3,             # 3rd Order (17 loudspeakers, 16 channels)
            normalization="sn3d" # SN3D format
        )

        print(f"Success! Saved to {final_out_path}")

    print("\n--- All HOA Processing Complete! ---")


if __name__ == "__main__":
    main()