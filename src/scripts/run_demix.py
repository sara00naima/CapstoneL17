import sys
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.demix import demix_folder


def main():
    input_folder = PROJECT_ROOT / "Demixing BS-RoF" / "songs"
    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"

    # Aggiorna questo path in base a dove tieni il checkpoint .ckpt
    model_path = PROJECT_ROOT / "Demixing BS-RoF" / "bs_roformer" / "models" / "BS-Rofo-SW-Fixed.ckpt"

    print("--- THE DEMIXING STAGE ---")

    demix_folder(
        input_dir=str(input_folder),
        output_dir=str(output_folder),
        model_path=str(model_path),
    )

    print("\n--- Demixing Complete! ---")


if __name__ == "__main__":
    main()