import sys
from collections import defaultdict
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import encode_stems_to_foa

STEM_TYPES = ["vocals", "drums", "bass", "guitar", "piano", "other"]

DEFAULT_POSITIONS_DEG = {
    "vocals": (0.0, 0.0),
    "drums": (180.0, 0.0),
    "bass": (0.0, -20.0),
    "guitar": (-45.0, 0.0),
    "piano": (45.0, 0.0),
    "other": (0.0, 60.0),
}


def collect_stems_by_song(output_folder: Path) -> dict[str, dict[str, str]]:
    all_songs_stems = defaultdict(dict)

    for wav_path in output_folder.glob("*.wav"):
        if wav_path.stem.endswith("_3d_scene"):
            continue

        for stem in STEM_TYPES:
            suffix = f"_{stem}"
            if wav_path.stem.endswith(suffix):
                song_name = wav_path.stem[: -len(suffix)]
                all_songs_stems[song_name][stem] = str(wav_path)
                break

    return dict(all_songs_stems)


def main():
    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    print("--- THE POSITIONING & ENCODING STAGE ---")

    all_songs_stems = collect_stems_by_song(output_folder)
    if not all_songs_stems:
        print(f"No stems found in {output_folder}")
        return

    print("Assigning 3D coordinates...")
    print("\nStarting Ambisonic Encoding...")

    for song_name, stem_paths in all_songs_stems.items():
        missing_stems = [stem for stem in STEM_TYPES if stem not in stem_paths]
        if missing_stems:
            print(f"Warning: '{song_name}' is missing stems {missing_stems}. Skipping.")
            continue

        final_out_path = output_folder / f"{song_name}_3d_scene.wav"
        print(f"Building 3D scene for: {song_name}")

        encode_stems_to_foa(
            stem_paths=stem_paths,
            positions_deg=DEFAULT_POSITIONS_DEG,
            out_path=str(final_out_path),
            convention="basic",
        )

        print(f"Success! Saved to {final_out_path}")

    print("\n--- All Processing Complete! ---")


if __name__ == "__main__":
    main()