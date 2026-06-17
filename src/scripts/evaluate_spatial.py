"""
evaluate_spatial.py
===================
Evaluation tools for the spatial audio pipeline.

Provides:
  1. plot_speaker_energy_polar  — polar plot of per-speaker energy for layout-decoded files
  2. compute_itd_ild            — ITD and ILD estimation from a binaural WAV
  3. run_evaluation             — batch evaluation across all rendered files

Usage (from project root):
    python src/scripts/evaluate_spatial.py

Input folder: outputs/rendered/ (written by the GUI's "GENERATE" step)
  Both binaural and layout-decoded files live in this same folder.
  A file is treated as binaural if its name ends in "_binaural.wav"
  (this suffix is always enforced by gui_backend.py); everything else
  is treated as a layout-decoded, multichannel file for the polar plot.

Output folder: outputs/evaluation/
  ├── polar/      ← one PNG per test case (speaker energy polar plot)
  └── itd_ild/    ← one PNG per test case (ITD and ILD bar charts)
      itd_ild_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate
import csv

#Path setup
SCRIPT_DIR   = Path(__file__).resolve().parent
SRC_DIR      = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.config import (
    DEFAULT_RENDERED_DIR,
    MEASUREMENTS_CSV,
)
from spatial_pipeline.ambisonics.layout.speaker_layout import load_speaker_layout

#Output dirs
EVAL_DIR         = PROJECT_ROOT / "outputs" / "evaluation"
POLAR_DIR        = EVAL_DIR / "polar"
ITD_ILD_DIR      = EVAL_DIR / "itd_ild"

# Marker that identifies a rendered file as binaural. The GUI backend
# (gui_backend.py) always appends this suffix to binaural output filenames,
# even when the user supplies a custom name, so this is a reliable way to
# tell binaural (stereo) renders apart from layout-decoded (multichannel)
# renders sitting in the same outputs/rendered/ folder.
BINAURAL_SUFFIX = "_binaural"

# Expected DOA for each static test case (azimuth_deg, elevation_deg)
# Must match TEST_POSITIONS in generate_decoder_test_files.py
EXPECTED_DOA = {
    "all_front": (0.0,   0.0),
    "all_left":  (90.0,  0.0),
    "all_right": (-90.0, 0.0),
    "all_back":  (180.0, 0.0),
}

# 1. POLAR PLOT — per-speaker energy from LS17 decoded file

def plot_speaker_energy_polar(
    ls17_wav_path: str | Path,
    speakers,
    title: str = "",
    out_path: str | Path | None = None,
    expected_doa: tuple[float, float] | None = None,
) -> None:
    """
    Reads a multichannel decoded WAV and plots per-speaker RMS energy on a polar axis.

    Parameters
    ----------
    ls17_wav_path   : path to the layout-decoded WAV file (one channel per speaker)
    speakers        : list of Speaker objects from load_speaker_layout()
    title           : plot title
    out_path        : if given, saves the figure there; otherwise shows interactively
    expected_doa    : (azimuth_deg, elevation_deg) of the expected source direction
                      drawn as a red arrow on the plot
    """
    audio, sr = sf.read(str(ls17_wav_path))  # (samples, n_speakers)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    n_speakers = min(audio.shape[1], len(speakers))

    # RMS energy per channel (in dB, floor at -60 dB)
    rms = np.sqrt(np.mean(audio[:, :n_speakers] ** 2, axis=0))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    rms_db_norm = rms_db - rms_db.max()  # normalize to 0 dB peak

    # Separate horizontal speakers (elevation ≈ 0) and elevated ones
    azimuths  = np.array([s.azimuth_deg  for s in speakers[:n_speakers]])
    elevs     = np.array([s.elevation_deg for s in speakers[:n_speakers]])
    labels    = [s.label for s in speakers[:n_speakers]]

    # Convert azimuth to math angle (CCW from East → CCW from North front=top)
    # Our convention: 0=front, 90=left → for polar plot: theta=90-azimuth in degrees
    theta_rad = np.deg2rad(90.0 - azimuths)

    # Energy radius: map [-60..0] dB → [0..1]
    r = np.clip((rms_db_norm + 60) / 60, 0, 1)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")   # 0° (front) at top
    ax.set_theta_direction(-1)         # clockwise (left=90° on the left side)

    # Draw each speaker as a dot, size and color proportional to energy
    sc = ax.scatter(
        theta_rad, r,
        c=rms_db_norm, cmap="plasma",
        vmin=-30, vmax=0,
        s=120, zorder=3,
    )

    # Label each speaker
    for i, (th, ri, lbl, elv) in enumerate(zip(theta_rad, r, labels, elevs)):
        offset = 0.08
        ax.annotate(
            f"{lbl}\n{elv:+.0f}°el",
            xy=(th, ri),
            xytext=(th, ri + offset),
            ha="center", va="bottom",
            fontsize=7,
            color="white" if fig.get_facecolor()[0] < 0.5 else "black",
        )

    # Draw lines from center to each speaker
    for th, ri in zip(theta_rad, r):
        ax.plot([th, th], [0, ri], color="gray", lw=0.8, alpha=0.5)

    # Expected DOA arrow
    if expected_doa is not None:
        exp_az, exp_el = expected_doa
        exp_theta = np.deg2rad(90.0 - exp_az)
        ax.annotate(
            "", xy=(exp_theta, 0.95), xytext=(exp_theta, 0.0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )
        ax.text(
            exp_theta, 1.05, "expected",
            ha="center", va="center", color="red", fontsize=8, fontweight="bold",
        )

    plt.colorbar(sc, ax=ax, label="Energy (dB, relative)", pad=0.1, shrink=0.7)
    ax.set_title(title, pad=20, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["-45 dB", "-30 dB", "-15 dB", "0 dB"], fontsize=7)

    plt.tight_layout()
    if out_path:
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# 2. ITD / ILD — from binaural WAV

def compute_itd_ild(
    binaural_wav_path: str | Path,
    max_itd_ms: float = 1.0,
) -> dict:
    """
    Estimates ITD and ILD from a binaural stereo WAV.

    ITD: peak of the cross-correlation between L and R channels,
         restricted to ±max_itd_ms (avoids spurious peaks far from center).
         Positive ITD → sound arrives at LEFT ear first → source is on the LEFT.

    ILD: broadband level difference L - R in dB.
         Positive ILD → left ear is louder → source is on the LEFT.

    Returns a dict with keys:
        itd_ms      — Interaural Time Difference in milliseconds
        ild_db      — Interaural Level Difference in dB (broadband)
        ild_db_bands — ILD per octave band (dict: band_center_hz → ild_db)
        sample_rate
    """
    audio, sr = sf.read(str(binaural_wav_path))
    if audio.ndim == 1 or audio.shape[1] < 2:
        raise ValueError(f"Expected stereo file, got shape {audio.shape}")

    left  = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)

    #ITD via cross-correlation
    max_lag_samples = int(max_itd_ms * 1e-3 * sr)
    xcorr = correlate(left, right, mode="full")
    lags  = np.arange(-(len(left) - 1), len(left))

    # Restrict to ±max_itd_ms window
    mask  = np.abs(lags) <= max_lag_samples
    peak  = np.argmax(np.abs(xcorr[mask]))
    itd_samples = lags[mask][peak]
    itd_ms = itd_samples / sr * 1000.0

    #ILD broadband
    rms_left  = np.sqrt(np.mean(left  ** 2))
    rms_right = np.sqrt(np.mean(right ** 2))
    eps = 1e-12
    ild_db = 20 * np.log10((rms_left + eps) / (rms_right + eps))

    #ILD per octave band
    from scipy.signal import butter, sosfilt

    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    ild_db_bands = {}

    for fc in octave_bands:
        low  = fc / np.sqrt(2)
        high = fc * np.sqrt(2)
        if high >= sr / 2:
            continue
        sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
        l_f = sosfilt(sos, left)
        r_f = sosfilt(sos, right)
        rms_l = np.sqrt(np.mean(l_f ** 2))
        rms_r = np.sqrt(np.mean(r_f ** 2))
        ild_db_bands[fc] = 20 * np.log10((rms_l + eps) / (rms_r + eps))

    return {
        "itd_ms":       itd_ms,
        "ild_db":       ild_db,
        "ild_db_bands": ild_db_bands,
        "sample_rate":  sr,
    }


def plot_itd_ild(
    results: dict,
    title: str = "",
    expected_doa: tuple[float, float] | None = None,
    out_path: str | Path | None = None,
) -> None:
    """
    Plots ITD (single value) and ILD per octave band as bar charts.
    """
    itd_ms      = results["itd_ms"]
    ild_db      = results["ild_db"]
    ild_bands   = results["ild_db_bands"]

    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.35)

    #ITD
    ax1 = fig.add_subplot(gs[0])
    color = "steelblue" if itd_ms >= 0 else "coral"
    ax1.bar(["ITD"], [itd_ms], color=color, width=0.4)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("ms  (+ = left ear first)")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_title("Interaural Time Difference")
    ax1.text(0, itd_ms + 0.05 * np.sign(itd_ms + 1e-9),
             f"{itd_ms:+.3f} ms", ha="center", va="bottom", fontsize=9)

    #ILD per band
    ax2 = fig.add_subplot(gs[1])
    bands = sorted(ild_bands.keys())
    vals  = [ild_bands[b] for b in bands]
    x     = np.arange(len(bands))
    colors = ["steelblue" if v >= 0 else "coral" for v in vals]
    bars  = ax2.bar(x, vals, color=colors, width=0.6)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b} Hz" for b in bands], rotation=30, ha="right")
    ax2.set_ylabel("dB  (+ = left louder)")
    ax2.set_title(f"ILD per octave band  (broadband: {ild_db:+.2f} dB)")
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.15 * np.sign(val + 1e-9),
            f"{val:+.1f}", ha="center", va="bottom", fontsize=7,
        )

    # Expected DOA annotation
    if expected_doa is not None:
        exp_az = expected_doa[0]
        side   = "left" if exp_az > 0 else ("right" if exp_az < 0 else "front/back")
        fig.text(
            0.5, 0.97,
            f"Expected: az={exp_az:+.0f}° ({side})  →  ITD>0 & ILD>0 if left, <0 if right, ≈0 if front/back",
            ha="center", va="top", fontsize=8, color="darkred",
        )

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()

    if out_path:
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# 3. BATCH EVALUATION

def run_evaluation(
    rendered_dir:  Path = DEFAULT_RENDERED_DIR,
    polar_out:     Path = POLAR_DIR,
    itd_ild_out:   Path = ITD_ILD_DIR,
) -> None:
    """
    Scans rendered_dir for rendered WAV files and runs the full evaluation.

    Files ending in "_binaural.wav" are treated as binaural stereo renders
    (ITD/ILD analysis). All other WAV files are treated as layout-decoded,
    multichannel renders (per-speaker polar energy plot).

    Saves all figures and a CSV summary.
    """
    polar_out.mkdir(parents=True, exist_ok=True)
    itd_ild_out.mkdir(parents=True, exist_ok=True)

    speakers = load_speaker_layout(MEASUREMENTS_CSV)

    summary_rows = []

    if not rendered_dir.exists():
        print(f"[evaluate_spatial] Rendered folder not found: {rendered_dir}")
        print("Run the GUI's GENERATE step first, or pass a different --rendered-dir.")
        return

    all_wavs = sorted(rendered_dir.glob("*.wav"))
    layout_files   = [w for w in all_wavs if not w.stem.lower().endswith(BINAURAL_SUFFIX)]
    binaural_files = [w for w in all_wavs if w.stem.lower().endswith(BINAURAL_SUFFIX)]

    #Polar plots from layout-decoded files
    if not layout_files:
        print(f"[polar] No layout-decoded WAV files found in {rendered_dir}")
    else:
        print(f"\n[polar] Found {len(layout_files)} layout-decoded files")

    for wav in layout_files:
        stem = wav.stem
        test_name = next(
            (k for k in EXPECTED_DOA if stem.endswith(k)),
            None,
        )
        expected = EXPECTED_DOA.get(test_name)
        title = f"Speaker energy — {stem}"
        out   = polar_out / f"{stem}_polar.png"
        print(f"  {wav.name} → {out.name}")
        plot_speaker_energy_polar(wav, speakers, title=title,
                                  out_path=out, expected_doa=expected)

    #ITD / ILD from binaural files
    if not binaural_files:
        print(f"[itd/ild] No binaural WAV files found in {rendered_dir}")
    else:
        print(f"\n[itd/ild] Found {len(binaural_files)} binaural files")

    for wav in binaural_files:
        stem = wav.stem[: -len(BINAURAL_SUFFIX)]  # strip "_binaural"
        test_name = next(
            (k for k in EXPECTED_DOA if stem.endswith(k)),
            None,
        )
        expected = EXPECTED_DOA.get(test_name)

        print(f"  {wav.name}")
        try:
            res = compute_itd_ild(wav)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        title = f"ITD / ILD — {wav.stem}"
        out   = itd_ild_out / f"{wav.stem}_itd_ild.png"
        plot_itd_ild(res, title=title, expected_doa=expected, out_path=out)

        row = {
            "file":        wav.name,
            "test_case":   test_name or "unknown",
            "expected_az": expected[0] if expected else "",
            "expected_el": expected[1] if expected else "",
            "itd_ms":      f"{res['itd_ms']:+.4f}",
            "ild_db":      f"{res['ild_db']:+.4f}",
        }
        for fc, val in res["ild_db_bands"].items():
            row[f"ild_{fc}hz"] = f"{val:+.4f}"
        summary_rows.append(row)
        print(f"    ITD={res['itd_ms']:+.3f} ms   ILD={res['ild_db']:+.2f} dB")

    #CSV summary
    if summary_rows:
        csv_path = itd_ild_out / "itd_ild_summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved to {csv_path}")

    print("\nEvaluation complete.")
    print(f"  Polar plots : {polar_out}")
    print(f"  ITD/ILD     : {itd_ild_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rendered-dir",
        type=Path,
        default=DEFAULT_RENDERED_DIR,
        help="Folder containing rendered WAV files (default: outputs/rendered/)",
    )
    args = parser.parse_args()

    run_evaluation(rendered_dir=args.rendered_dir)