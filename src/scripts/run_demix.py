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

from spatial_pipeline.demix import demix_folder


def main():
    # Folder containing the raw stereo song files to be demixed (.wav)
    input_folder = PROJECT_ROOT / "Demixing BS-RoF" / "songs"

    # Folder where the separated stems will be written
    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    # Path to the pre-trained BS-RoFormer checkpoint
    model_path = PROJECT_ROOT / "Demixing BS-RoF" / "bs_roformer" / "models" / "roformer-model-bs-roformer-sw-by-jarredou" / "BS-Rofo-SW-Fixed.ckpt"

    print("--- THE DEMIXING STAGE ---")

    # Run source separation on all .wav files found in input_folder
    demix_folder(
        input_dir=str(input_folder),
        output_dir=str(output_folder),
        model_path=str(model_path),
    )

    print("\n--- Demixing Complete! ---")


if __name__ == "__main__":
    main()