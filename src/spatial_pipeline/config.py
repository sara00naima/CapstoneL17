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

# Test file output directories (used by generate_decoder_test_files.py).
# These are also the playback feeds referenced by the Reaper project
# Cremona_Test_Cases_test_audio.rpp, so the location must stay stable.
DEFAULT_TEST_DIR            = DEFAULT_OUTPUT_DIR / "test"
DEFAULT_TEST_HOA_DIR        = DEFAULT_TEST_DIR / "hoa"
DEFAULT_TEST_LS17_DIR       = DEFAULT_TEST_DIR / "ls17"
DEFAULT_TEST_BINAURAL_DIR   = DEFAULT_TEST_DIR / "binaural"
DEFAULT_TEST_LS17_BINAURAL_DIR = DEFAULT_TEST_DIR / "ls17_binaural"

# Evaluation outputs. Single root so every evaluation artifact lives together:
#   evaluate_spatial.py               -> polar/ and itd_ild/ (and re-plots the reproduction CSV)
#   evaluate_pipeline_reproduction.py -> pipeline_reproduction/pipeline_reproduction_eval.csv
DEFAULT_EVAL_DIR                   = DEFAULT_OUTPUT_DIR / "eval"
DEFAULT_POLAR_DIR                  = DEFAULT_EVAL_DIR / "polar"
DEFAULT_ITD_ILD_DIR                = DEFAULT_EVAL_DIR / "itd_ild"
DEFAULT_PIPELINE_REPRODUCTION_DIR  = DEFAULT_EVAL_DIR / "pipeline_reproduction"
DEFAULT_PIPELINE_REPRODUCTION_CSV  = DEFAULT_PIPELINE_REPRODUCTION_DIR / "pipeline_reproduction_eval.csv"

# Path to the default SOFA HRTF file used for binaural rendering
DEFAULT_HRTF_SOFA = PROJECT_ROOT / "hrtf" / "D1_44K_16bit_0.3s_FIR_SOFA.sofa"

