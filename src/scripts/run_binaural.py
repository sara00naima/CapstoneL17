import sys
from pathlib import Path

# Resolve directory structure relative to this script's location
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Add src/ to the Python path so that spatial_pipeline can be imported
# as a package without needing a formal install
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import render_binaural_scene
from spatial_pipeline.config import DEFAULT_HRTF_SOFA, DEFAULT_BINAURAL_DIR


def main():
    print("--- BINAURAL RENDERING STAGE ---")

    # Folder where the HOA encoding stage wrote the scene files
    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    # Pick up all HOA scene files produced by run_encode_hoa.py
    scene_files = list(output_folder.glob("*_hoa3.wav"))

    if not scene_files:
        print("No HOA scene files found. Run run_encode_hoa.py first.")
        return

    if not DEFAULT_HRTF_SOFA.exists():
        print(f"SOFA file not found at: {DEFAULT_HRTF_SOFA}")
        print("Place your SOFA HRTF file there, or update DEFAULT_HRTF_SOFA in config.py.")
        return

    # Create the binaural output folder if it doesn't exist yet
    DEFAULT_BINAURAL_DIR.mkdir(parents=True, exist_ok=True)

    for scene_path in scene_files:
        # Recover the original song name from the scene filename
        song_name = scene_path.stem.replace("_hoa3", "")
        print(f"\nRendering binaural mix for: {song_name}")

        # Output path for the stereo binaural file
        out_path = DEFAULT_BINAURAL_DIR / f"{song_name}_binaural.wav"

        render_binaural_scene(
            scene_path=str(scene_path),
            sofa_path=str(DEFAULT_HRTF_SOFA),
            out_path=str(out_path),
            order=3,
        )

        print(f"Saved to: {out_path}")

    print("\n--- All Binaural Rendering Complete! ---")


if __name__ == "__main__":
    main()
