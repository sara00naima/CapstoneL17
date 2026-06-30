#!/usr/bin/env python3
"""
doa_table_pseudospectrum.py
===========================
Report artifacts for the Results section: a per-test DOA table and a REAL-vs-SIM
pseudospectrum figure, for the four STATIC cardinal pans across the three mixes.

The "predicted DOA" is read from the IDEAL anechoic decode (SIM) -- i.e. it
validates the *decoder* (where it aims), which is presentable (~10-30 deg),
NOT the raw EM32 recording (whose central field is near-diffuse, so its argmax
DOA is meaningless ~150 deg).  The physical recording enters only as a separate
REAL-vs-SIM map-correlation column (reproduction-fidelity indicator).

Reuses the simulate-and-compare machinery in evaluate_pipeline_reproduction.py.

Usage
-----
    python src/scripts/doa_table_pseudospectrum.py            # diagnostic only
    python src/scripts/doa_table_pseudospectrum.py --latex --figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import estimate_speaker_positions as esp                      # noqa: E402
import evaluate_pipeline_reproduction as repro                # noqa: E402

SRC_DIR = SCRIPTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from spatial_pipeline.config import DEFAULT_PIPELINE_REPRODUCTION_DIR  # noqa: E402

SONGS = ["garage", "message", "test"]
STATIC = ["front", "back", "left", "right"]


# --------------------------------------------------------------------------- #
# Beamformers (mic-frame analysis grid + room-frame plotting grid)
# --------------------------------------------------------------------------- #
def dome_bf(fs: int, band: tuple[float, float]) -> "esp.Beamformer":
    """Dome search grid (el >= EL_FLOOR) in the MIC frame, custom band."""
    dirs = esp.grid_directions(repro.GRID_STEP_DEG)
    dirs = dirs[dirs[:, 2] >= np.sin(np.deg2rad(repro.EL_FLOOR_DEG))]
    return esp.Beamformer(dirs, esp.SH_ORDER, band, repro.WIN_LEN, fs)


def room_grid_bf(fs: int, band: tuple[float, float], R: np.ndarray,
                 az_step: float = 3.0, el_step: float = 3.0,
                 el_lo: float = -30.0):
    """Regular (az, el) grid in the ROOM/ambisonic frame for a clean heatmap.

    Returns (bf, azs, els): bf scores room directions (converted to the mic
    frame via dir_mic = dir_room @ R), so M reshapes to (len(els), len(azs)).
    """
    azs = np.arange(-180.0, 180.0 + 1e-6, az_step)
    els = np.arange(el_lo, 90.0 + 1e-6, el_step)
    A, E = np.meshgrid(np.deg2rad(azs), np.deg2rad(els))
    a, e = A.ravel(), E.ravel()
    dir_room = np.column_stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    dir_mic = dir_room @ R                                    # room -> mic
    return esp.Beamformer(dir_mic, esp.SH_ORDER, band, repro.WIN_LEN, fs), azs, els


# --------------------------------------------------------------------------- #
# DOA readouts from a power map
# --------------------------------------------------------------------------- #
def map_argmax(M: np.ndarray, bf, R: np.ndarray):
    az, el = repro.vec_to_amb(R @ bf.dirs[int(np.argmax(M))])
    return az, el


def map_centroid(M: np.ndarray, bf, R: np.ndarray, keep_pct: float = 10.0):
    """Energy-weighted centroid of the top-`keep_pct`% of the map (robust DOA)."""
    dir_room = bf.dirs @ R.T                                  # mic -> room
    thr = np.percentile(M, 100.0 - keep_pct)
    sel = M >= thr
    w = M[sel]
    v = (w[:, None] * dir_room[sel]).sum(axis=0)
    return repro.vec_to_amb(v)


def az_err(pred_az: float, pan_az: float) -> float:
    """Wrapped absolute azimuth error in degrees."""
    return abs((pred_az - pan_az + 180.0) % 360.0 - 180.0)


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #
def rec_path(recs_dir: Path, song: str, test: str) -> Path | None:
    hits = list((recs_dir / f"{song}_recs").glob(f"*_{test}.wav"))
    return hits[0] if hits else None


def evaluate(recs_dir, media_dir, bf, A, prop, R, keep_pct):
    rows = []
    for song in SONGS:
        for test in STATIC:
            pan = repro.TEST_SPECS[test][2]
            suffix = repro.TEST_SPECS[test][1]
            rp = rec_path(recs_dir, song, test)
            feed = repro.find_media(media_dir, song, suffix)
            if rp is None or feed is None:
                print(f"  [skip] {song} {test}: rec={rp} feed={feed}")
                continue
            Mr, _, _ = repro.real_response(rp, bf)
            Ms, _, _ = repro.sim_response(feed, bf, A, prop)
            corr = float(np.corrcoef(Mr, Ms)[0, 1])
            caz, cel = map_centroid(Ms, bf, R, keep_pct)
            aaz, ael = map_argmax(Ms, bf, R)
            err_c = repro.ang_between(repro.amb_vec(caz, cel), repro.amb_vec(*pan))
            err_a = repro.ang_between(repro.amb_vec(aaz, ael), repro.amb_vec(*pan))
            rows.append({
                "song": song, "test": test, "pan_az": pan[0], "pan_el": pan[1],
                "sim_az": caz, "sim_el": cel, "err": err_c,
                "az_err": az_err(caz, pan[0]),
                "argmax_az": aaz, "argmax_el": ael, "argmax_err": err_a,
                "argmax_az_err": az_err(aaz, pan[0]),
                "sim_pkmean": float(Ms.max() / Ms.mean()),
                "real_pkmean": float(Mr.max() / Mr.mean()),
                "corr": corr,
            })
    return rows


def print_diag(rows, band):
    print(f"\n=== DIAGNOSTIC  band {band[0]:.0f}-{band[1]:.0f} Hz ===")
    print(f"{'song':<8}{'test':<7}{'pan':>10}{'SIM argmax':>14}"
          f"{'3Derr':>6}{'azerr':>6}{'pk/mnS':>7}{'pk/mnR':>7}{'corr':>7}")
    for r in rows:
        print(f"{r['song']:<8}{r['test']:<7}"
              f"({r['pan_az']:+.0f},{r['pan_el']:+.0f})".rjust(10)
              + f"({r['argmax_az']:+.0f},{r['argmax_el']:+.0f})".rjust(14)
              + f"{r['argmax_err']:>6.0f}{r['argmax_az_err']:>6.0f}"
              + f"{r['sim_pkmean']:>7.2f}{r['real_pkmean']:>7.2f}{r['corr']:>+7.2f}")
    # per-test means (argmax)
    print("  -- per-test mean: 3D err / az err / corr  (argmax) --")
    for test in STATIC:
        sub = [r for r in rows if r["test"] == test]
        if sub:
            print(f"    {test:<7} 3D {np.mean([r['argmax_err'] for r in sub]):5.1f}"
                  f"   az {np.mean([r['argmax_az_err'] for r in sub]):5.1f}"
                  f"   corr {np.mean([r['corr'] for r in sub]):+.2f}")


# --------------------------------------------------------------------------- #
# LaTeX table
# --------------------------------------------------------------------------- #
def write_latex(rows, path: Path):
    L = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Decoder aim for the four static pans, read from the ideal"
        r" anechoic decode (SIM) beamformed identically to the measurements, with"
        r" the REAL-vs-SIM map correlation as a reproduction-fidelity indicator.}",
        r"  \label{tab:doa}",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    Test & Mix & Intended DOA & Predicted DOA (SIM) & Az.\ err & 3D err & Map-corr \\",
        r"         &     & (az, el)     & (az, el)            & (deg)    & (deg)  & (real)   \\",
        r"    \midrule",
    ]
    for test in STATIC:
        sub = [r for r in rows if r["test"] == test]
        for i, r in enumerate(sub):
            name = test.capitalize() if i == 0 else ""
            L.append(
                f"    {name} & {r['song']} & "
                f"$({r['pan_az']:+.0f},\\,{r['pan_el']:+.0f})$ & "
                f"$({r['argmax_az']:+.0f},\\,{r['argmax_el']:+.0f})$ & "
                f"{r['argmax_az_err']:.0f} & {r['argmax_err']:.0f} & {r['corr']:+.2f} \\\\"
            )
        if sub:
            maz = np.mean([r["argmax_az_err"] for r in sub])
            m3d = np.mean([r["argmax_err"] for r in sub])
            mc = np.mean([r["corr"] for r in sub])
            L.append(f"    & \\textbf{{mean}} & & & \\textbf{{{maz:.0f}}} & "
                     f"\\textbf{{{m3d:.0f}}} & \\textbf{{{mc:+.2f}}} \\\\")
            L.append(r"    \midrule")
    if L[-1].strip() == r"\midrule":
        L[-1] = r"    \bottomrule"
    else:
        L.append(r"    \bottomrule")
    L += [r"  \end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nLaTeX table -> {path}")


# --------------------------------------------------------------------------- #
# Pseudospectrum figure (REAL vs SIM, room frame)
# --------------------------------------------------------------------------- #
def make_figure(recs_dir, media_dir, R, fs, band, A_unused, prop_unused,
                song, test, out_path: Path, layout_csv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bf, azs, els = room_grid_bf(fs, band, R)
    A, prop, _ = repro.build_speaker_model(bf, R, layout_csv, fs)
    pan = repro.TEST_SPECS[test][2]
    suffix = repro.TEST_SPECS[test][1]
    rp = rec_path(recs_dir, song, test)
    feed = repro.find_media(media_dir, song, suffix)

    Mr = repro.real_response(rp, bf)[0].reshape(len(els), len(azs))
    Ms = repro.sim_response(feed, bf, A, prop)[0].reshape(len(els), len(azs))

    def to_db(M):
        m = M / M.max()
        return 10.0 * np.log10(np.maximum(m, 1e-3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    extent = [azs[0], azs[-1], els[0], els[-1]]
    for ax, M, ttl in ((axes[0], Ms, "SIM (ideal decode)"),
                       (axes[1], Mr, "REAL (EM32 recording)")):
        im = ax.imshow(to_db(M), origin="lower", extent=extent, aspect="auto",
                       cmap="inferno", vmin=-10, vmax=0)
        ax.plot(pan[0], pan[1], "o", mfc="none", mec="cyan", mew=2, ms=14)
        ax.axhline(0, color="white", lw=0.5, alpha=0.3)
        ax.set_xlabel("azimuth (deg)  [0 = front, +90 = left]")
        ax.set_title(f"{ttl}   pk/mean {M.max()/M.mean():.1f}")
        ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel("elevation (deg)")
    fig.colorbar(im, ax=axes, label="normalised power (dB)", pad=0.02, shrink=0.85)
    fig.suptitle(f"Beamformer pseudospectrum — {song} / {test} "
                 f"(o = intended {pan})", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPseudospectrum figure -> {out_path}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recs-dir", type=Path, default=repro.DEFAULT_RECS_DIR)
    ap.add_argument("--media-dir", type=Path, default=repro.DEFAULT_MEDIA_DIR)
    ap.add_argument("--layout-csv", type=Path, default=repro.DEFAULT_LAYOUT_CSV)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--band", type=float, nargs=2, default=[400.0, 5000.0])
    ap.add_argument("--keep-pct", type=float, default=10.0)
    ap.add_argument("--rotation-cache", type=Path, default=SCRIPTS_DIR / ".doa_R.npy")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--fig-song", default="garage")
    ap.add_argument("--fig-test", default="front")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_PIPELINE_REPRODUCTION_DIR)
    args = ap.parse_args()
    band = tuple(args.band)

    print("Step 1: mic->room rotation R from the sweep recordings")
    if args.rotation_cache.is_file() and not args.recompute:
        R, fs = np.load(args.rotation_cache), 48000
        print(f"  loaded cached R from {args.rotation_cache}")
    else:
        R, fs = repro.compute_mic_to_room(args.reg)
        np.save(args.rotation_cache, R)
        print(f"  saved R to {args.rotation_cache}")

    bf = dome_bf(fs, band)
    A, prop, _ = repro.build_speaker_model(bf, R, args.layout_csv, fs)

    print(f"\nStep 2: evaluate 4 static pans x 3 mixes (band {band[0]:.0f}-{band[1]:.0f} Hz, "
          f"keep top {args.keep_pct:.0f}%)")
    rows = evaluate(args.recs_dir, args.media_dir, bf, A, prop, R, args.keep_pct)
    print_diag(rows, band)

    if args.latex:
        write_latex(rows, args.out_dir / "doa_table.tex")
    if args.figure:
        make_figure(args.recs_dir, args.media_dir, R, fs, band, A, prop,
                    args.fig_song, args.fig_test,
                    args.out_dir / f"pseudospectrum_{args.fig_song}_{args.fig_test}.png",
                    args.layout_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
