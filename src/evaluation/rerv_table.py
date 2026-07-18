#!/usr/bin/env python3
"""
rerv_table.py
=============
Measurement-independent decoder check for the Results chapter: the Gerzon
energy (rE) and velocity (rV) localisation vectors of the third-order
pseudo-inverse LS17 decoder, for the four static cardinal pans.

Why this table exists
---------------------
In the simulate-and-compare analysis (evaluate_pipeline_reproduction.py) the
mic->room rotation R cancels algebraically in the SIM path, so "SIM recovers the
pan" is partly true by construction and cannot, on its own, prove the decoder is
good.  rE/rV validate the decoder analytically -- no microphone, no beamformer,
no recording -- straight from the speaker gains the decoder produces.

  rV = sum_i g_i u_i / sum_i g_i        (velocity / low-frequency localisation)
  rE = sum_i g_i^2 u_i / sum_i g_i^2    (energy   / mid-high localisation)

where u_i is the unit direction of loudspeaker i and g_i is the gain the decoder
sends to it for a source panned to the target direction.  |rV|,|rE| <= 1 measure
how concentrated (well-localised) the reproduced field is; their directions are
where the decoder tells the ear the source is.

Decoder
-------
Same directional decoder the pipeline uses: D = pinv(Y), Y = real SH basis
(ACN/SN3D, order 3) sampled at the measured loudspeaker directions of
measurements_transcription.csv.  The per-speaker distance-gain compensation from
calculate_decoder_matrix is deliberately NOT applied here: it is a reproduction
level trim, not part of the directional decode, and rE/rV are direction/energy
descriptors.  (With SN3D or N3D the vectors are identical, since the
normalisation cancels in both ratios.)

Reproduces outputs/eval/pipeline_reproduction/rE_rV_table.tex exactly.

Usage
-----
    python src/evaluation/rerv_table.py            # print + write the .tex
    python src/evaluation/rerv_table.py --no-latex # print only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.config import (                      # noqa: E402
    MEASUREMENTS_CSV,
    DEFAULT_PIPELINE_REPRODUCTION_DIR,
)
from spatial_pipeline.ambisonics.layout.speaker_layout import (  # noqa: E402
    load_speaker_layout,
)
from spatial_pipeline.ambisonics.core.spherical_harmonics import sh_basis_real  # noqa: E402

ORDER = 3
NORMALIZATION = "sn3d"

# name -> (azimuth, elevation) in the ambisonic frame (deg): 0 = front, +90 = left
PANS = [
    ("Front", (0.0, 0.0)),
    ("Back", (180.0, 0.0)),
    ("Left", (90.0, 0.0)),
    ("Right", (-90.0, 0.0)),
]


def vec_to_azel(v: np.ndarray) -> tuple[float, float]:
    """Cartesian (x=front, y=left, z=up) -> (azimuth, elevation) in degrees."""
    v = v / (np.linalg.norm(v) + 1e-12)
    az = np.rad2deg(np.arctan2(v[1], v[0]))
    el = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return float(az), float(el)


def directional_decoder(az_rad: np.ndarray, el_rad: np.ndarray) -> np.ndarray:
    """Pseudo-inverse directional decoder D (channels, speakers), no distance gain."""
    Y = np.stack(
        [sh_basis_real(ORDER, az_rad[i], el_rad[i], NORMALIZATION) for i in range(len(az_rad))],
        axis=0,
    )  # (speakers, channels)
    return np.linalg.pinv(Y)


def gerzon_vectors(gains: np.ndarray, dirs: np.ndarray):
    """Return (rV, rE) 3-vectors from speaker gains and unit directions."""
    rV = (gains[:, None] * dirs).sum(axis=0) / gains.sum()
    rE = ((gains ** 2)[:, None] * dirs).sum(axis=0) / (gains ** 2).sum()
    return rV, rE


def compute_rows():
    speakers = load_speaker_layout(MEASUREMENTS_CSV)
    az_rad = np.array([s.azimuth_rad for s in speakers])
    el_rad = np.array([s.elevation_rad for s in speakers])
    dirs = np.stack([s.unit_vector for s in speakers], axis=0)  # (K,3)
    D = directional_decoder(az_rad, el_rad)

    rows = []
    for name, (paz, pel) in PANS:
        y = sh_basis_real(ORDER, np.deg2rad(paz), np.deg2rad(pel), NORMALIZATION)
        g = y @ D                                              # (speakers,)
        rV, rE = gerzon_vectors(g, dirs)
        vaz, vel = vec_to_azel(rV)
        eaz, eel = vec_to_azel(rE)
        rows.append({
            "test": name, "pan_az": paz, "pan_el": pel,
            "rv_az": vaz, "rv_el": vel, "rv_mag": float(np.linalg.norm(rV)),
            "re_az": eaz, "re_el": eel, "re_mag": float(np.linalg.norm(rE)),
        })
    return rows


def print_diag(rows):
    print(f"\n=== rE / rV  (order {ORDER}, {NORMALIZATION}, pinv decoder, "
          f"{MEASUREMENTS_CSV.name}) ===")
    print(f"{'Test':<7}{'target':>12}{'rV dir':>12}{'|rV|':>7}{'rE dir':>12}{'|rE|':>7}")
    for r in rows:
        print(f"{r['test']:<7}"
              + f"({r['pan_az']:+.0f},{r['pan_el']:+.0f})".rjust(12)
              + f"({r['rv_az']:+.0f},{r['rv_el']:+.0f})".rjust(12)
              + f"{r['rv_mag']:>7.2f}"
              + f"({r['re_az']:+.0f},{r['re_el']:+.0f})".rjust(12)
              + f"{r['re_mag']:>7.2f}")


def write_latex(rows, path: Path):
    L = [
        r"\begin{table}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Measurement-independent decoder analysis: Gerzon energy ($r_E$)"
        r" and velocity ($r_V$) vectors of the third-order pseudo-inverse decoder"
        r" for the four static pans. Direction is the localisation predicted by the"
        r" decoder; $|r_E|,|r_V|\le1$ measure its concentration.}",
        r"\label{tab:rerv}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Test & Target & $r_V$ dir & $|r_V|$ & $r_E$ dir & $|r_E|$ \\",
        r"     & (az,el) & (az,el)   &         & (az,el)   &         \\",
        r"\midrule",
    ]
    for r in rows:
        L.append(
            f"{r['test']} & $({r['pan_az']:+.0f},{r['pan_el']:+.0f})$ & "
            f"$({r['rv_az']:+.0f},{r['rv_el']:+.0f})$ & {r['rv_mag']:.2f} & "
            f"$({r['re_az']:+.0f},{r['re_el']:+.0f})$ & {r['re_mag']:.2f} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nLaTeX table -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_PIPELINE_REPRODUCTION_DIR / "rE_rV_table.tex")
    ap.add_argument("--no-latex", action="store_true", help="print only, do not write the .tex")
    args = ap.parse_args()

    rows = compute_rows()
    print_diag(rows)
    if not args.no_latex:
        write_latex(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
