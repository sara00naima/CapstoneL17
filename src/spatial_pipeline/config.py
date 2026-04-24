from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent.parent

MEASUREMENTS_CSV = PROJECT_ROOT / "measurements_transcription.csv"

BSROFORMER_ROOT = PROJECT_ROOT / "Demixing BS-RoF" / "bs_roformer"
BSROFORMER_CONFIG = BSROFORMER_ROOT / "configs" / "config_bs_roformer_sw.yaml"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DEMIX_DIR = DEFAULT_OUTPUT_DIR / "demixed"
DEFAULT_FOA_DIR = DEFAULT_OUTPUT_DIR / "foa"