from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

MEASUREMENTS_CSV = PROJECT_ROOT / "measurements_transcription.csv"

BSROFORMER_ROOT = PROJECT_ROOT / "Demixing BS-RoF" / "bs_roformer"
BSROFORMER_CONFIG = BSROFORMER_ROOT / "configs" / "config_bs_roformer_sw.yaml"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DEMIX_DIR = DEFAULT_OUTPUT_DIR / "demixed"
DEFAULT_FOA_DIR = DEFAULT_OUTPUT_DIR / "foa"
DEFAULT_HOA_DIR = DEFAULT_OUTPUT_DIR / "hoa"
DEFAULT_LS17_DIR = DEFAULT_OUTPUT_DIR / "ls17"
DEFAULT_BINAURAL_DIR = DEFAULT_OUTPUT_DIR / "binaural"
DEFAULT_GUI_DIR = DEFAULT_OUTPUT_DIR / "gui"

# Folder where the GUI writes its rendered output (binaural and/or layout-decoded
# files), produced by _do_generate() in gui_backend.py. Both file types live
# together here; evaluate_spatial.py tells them apart by filename suffix
# (binaural files always end in "_binaural.wav", enforced by the GUI backend).
DEFAULT_RENDERED_DIR = DEFAULT_OUTPUT_DIR / "rendered"

# Test file output directories (used by generate_decoder_test_files.py)
DEFAULT_TEST_DIR            = DEFAULT_OUTPUT_DIR / "test"
DEFAULT_TEST_HOA_DIR        = DEFAULT_TEST_DIR / "hoa"
DEFAULT_TEST_LS17_DIR       = DEFAULT_TEST_DIR / "ls17"
DEFAULT_TEST_BINAURAL_DIR   = DEFAULT_TEST_DIR / "binaural"
DEFAULT_TEST_LS17_BINAURAL_DIR = DEFAULT_TEST_DIR / "ls17_binaural"

# Path to the default SOFA HRTF file used for binaural rendering
DEFAULT_HRTF_SOFA = PROJECT_ROOT / "hrtf" / "D1_44K_16bit_0.3s_FIR_SOFA.sofa"

