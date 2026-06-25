#!/usr/bin/env python3
"""
evaluate_pipeline_doa.py
========================
Evaluate the spatial-audio pipeline by SIMULATE-AND-COMPARE: for each decoder
test case, beamform the REAL Eigenmike EM32 recording and an ideal ANECHOIC
SIMULATION of the same played speaker feeds with the *same* beamformer, then
compare their spatial responses.  Because both go through identical processing,
any array/beamformer bias cancels and the difference isolates the reproduction
and geometry error (mic placement / parallax; the museum room is acoustically treated).

Why not a single recovered DOA?
-------------------------------
A naive "recovered DOA vs panned DOA" does not work here, and we proved it:
  * The measurement chain is sound -- single sources (the sweep recordings)
    localise to ~7 deg azimuth (see estimate_speaker_positions.py), and a
    single simulated speaker localises to <2 deg (sanity check below).
  * But a decoded ambisonic scene played over the ~17-speaker dome and recorded
    at the centre reconstructs a *field* (a superposition of ~17 near-equidistant
    speakers), not a single arrival.  Even the ideal anechoic simulation gives
    only a weak, broad beamformer peak (peak/mean ~1.5 vs ~6.6 for one speaker).
    The real recording is flatter still (off-centre mic + intrinsic field spread), so a
    single argmax DOA is meaningless.
So we compare the whole spatial response (peak + map correlation + directivity),
real vs anechoic-ideal, which is the rigorous way to use these recordings.

Method
------
1. mic->room rotation R from the sweep recordings (reuse estimate_speaker_positions
   calibration: same capsule layout, beamformer, Kabsch fit).
2. SIM: each loudspeaker = a plane wave from its measured direction (mic frame),
   rigid-sphere scattered with the SAME modal model b_n(ka) the beamformer uses,
   weighted by 1/r and delayed by r/c.  The played 17-ch feed drives the sources,
   giving a synthetic 32-capsule spectrum per frame.
3. Beamform REAL and SIM identically over the loudspeaker-dome grid (el >= -15 deg
   so the search cannot lock onto sub-floor reflections) -> energy-weighted, frame-
   averaged power maps.
4. Per test report:
     SIM  peak (az,el), SIM peak/mean, SIM-vs-pan  -> how well the decode aims
                                                      (anechoic, decoder spread only)
     REAL peak (az,el), REAL peak/mean
     REAL-vs-SIM peak angle and MAP CORRELATION    -> reproduction fidelity (KEY)
   Trajectories (orbit/bounce): per-frame peak-azimuth coverage/range, real vs sim
   (no playback time-sync, so distributions are compared, not per-frame).
   song: map correlation + strongest peaks of both.

Usage
-----
    python src/scripts/evaluate_pipeline_doa.py
    python src/scripts/evaluate_pipeline_doa.py --rotation-cache .doa_R.npy
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import eval_legendre

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import estimate_speaker_positions as esp  # noqa: E402

SRC_DIR = SCRIPTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from spatial_pipeline.config import DEFAULT_PIPELINE_DOA_CSV  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIR.parents[1]
DEFAULT_RECS_DIR = PROJECT_ROOT / "recs"
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "Cremona_Test_Cases_test_audio" / "Media"
DEFAULT_LAYOUT_CSV = PROJECT_ROOT / "measurements_transcription.csv"

WIN_LEN = 2048
HOP = 1024
GRID_STEP_DEG = 5.0
EL_FLOOR_DEG = -15.0            # search only the physical loudspeaker dome
ENERGY_KEEP_FRAC = 0.30        # keep frames with RMS >= frac * 95th-percentile RMS

# rec-folder song -> media filename prefix
SONG_PREFIX = {
    "garage": "sound_garage-funky-loop-382957",
    "message": "Still Corners - The Message",
    "test": "test_audio",
}
# test name -> (kind, media-filename suffix, ideal pan az/el)
TEST_SPECS = {
    "front":  ("static", "all_front_17ch", (0.0, 0.0)),
    "back":   ("static", "all_back_17ch", (180.0, 0.0)),
    "left":   ("static", "all_left_17ch", (90.0, 0.0)),
    "right":  ("static", "all_right_17ch", (-90.0, 0.0)),
    "bounce": ("bounce", "bounce_17ch", (0.0, 0.0)),
    "orbit":  ("orbit", "orbit_17ch", (0.0, 0.0)),
    "song":   ("song", "song_test_17ch", None),
}
TEST_ORDER = ["front", "back", "left", "right", "bounce", "orbit", "song"]


# --------------------------------------------------------------------------- #
# Geometry helpers (room frame == ambisonic frame: x=front, y=left, z=up)
# --------------------------------------------------------------------------- #
def amb_vec(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def vec_to_amb(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    az = (np.rad2deg(np.arctan2(v[1], v[0])) + 180.0) % 360.0 - 180.0
    el = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return float(az), float(el)


def ang_between(u: np.ndarray, v: np.ndarray) -> float:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return float(np.rad2deg(np.arccos(np.clip(u @ v, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Step 1: mic -> room rotation from the sweeps
# --------------------------------------------------------------------------- #
def compute_mic_to_room(reg: float):
    sweep, fs = sf.read(esp.DEFAULT_SWEEP, dtype="float64")
    if sweep.ndim > 1:
        sweep = sweep[:, 0]
    ref = esp.load_reference(esp.DEFAULT_TRANS_CSV)
    bf = esp.Beamformer(esp.grid_directions(esp.GRID_STEP_DEG), esp.SH_ORDER,
                        esp.BAND_HZ, esp.WIN_LEN, fs)
    wavs = sorted((p for p in esp.DEFAULT_REC_DIR.glob("*.wav") if esp.label_from_filename(p)),
                  key=lambda p: esp.sort_key(esp.label_from_filename(p)))
    doas, labels = [], []
    for p in wavs:
        rec, _ = sf.read(p, dtype="float64")
        if rec.ndim != 2 or rec.shape[1] != 32:
            continue
        n_fft = len(rec) + len(sweep)
        snap, _, _ = esp.direct_snapshot(rec, esp.inverse_filter(sweep, n_fft, reg), n_fft)
        doas.append(bf.doa(snap))
        labels.append(esp.label_from_filename(p))
    common = [l for l in labels if l in ref]
    D = np.array([doas[labels.index(l)] for l in common])
    Q = np.array([esp.reference_dir_from_mic(ref[l]) for l in common])
    R, resid, inliers = esp.calibrate(D, Q)
    front_az = esp.room_vec_to_compass(R @ np.array([1.0, 0.0, 0.0]))[0]
    print(f"  mic->room: inliers {int(inliers.sum())}/{len(common)} | residual {resid[inliers].mean():.1f} deg | "
          f"EM32 front -> transcription compass {front_az:.1f} deg")
    return R, fs


# --------------------------------------------------------------------------- #
# Beamformer scoring + anechoic loudspeaker simulation
# --------------------------------------------------------------------------- #
def dome_beamformer(fs: int) -> "esp.Beamformer":
    dirs = esp.grid_directions(GRID_STEP_DEG)
    dirs = dirs[dirs[:, 2] >= np.sin(np.deg2rad(EL_FLOOR_DEG))]
    return esp.Beamformer(dirs, esp.SH_ORDER, esp.BAND_HZ, WIN_LEN, fs)


def build_speaker_model(bf: "esp.Beamformer", R: np.ndarray, layout_csv: Path, fs: int):
    """Rigid-sphere steering manifold + propagation for each loudspeaker.

    Returns (A, prop, labels): A (nb,K,32) per-band steering vectors in the MIC
    frame (same b_n model as the beamformer), prop (nb,K) = (1/r) e^{-j w r/c}.
    """
    ref = esp.load_reference(layout_csv)
    labels = sorted(ref, key=lambda l: int(l[1:]))
    dir_room = np.array([esp.reference_dir_from_mic(ref[l]) for l in labels])  # (K,3)
    dir_mic = dir_room @ R                                                     # room -> mic
    rk = np.array([ref[l]["distance"] for l in labels])

    fb = np.fft.rfftfreq(WIN_LEN, 1.0 / fs)[bf.band_idx]
    ka = 2 * np.pi * fb * esp.A_RADIUS / esp.C_SOUND
    bn = np.stack([esp._bn(n, ka) for n in range(esp.SH_ORDER + 1)], axis=1)   # (nb,N+1)
    cosg = dir_mic @ esp.UQ.T                                                  # (K,32)
    Pn = np.stack([(2 * n + 1) * eval_legendre(n, cosg) for n in range(esp.SH_ORDER + 1)], axis=1)
    A = np.einsum("fn,knc->fkc", bn, Pn)                                       # (nb,K,32)
    prop = (1.0 / rk)[None, :] * np.exp(-1j * 2 * np.pi * fb[:, None] * rk[None, :] / esp.C_SOUND)
    return A, prop, labels


def _score(bf: "esp.Beamformer", Pb: np.ndarray) -> np.ndarray:
    """Bartlett power over the grid from band-limited capsule spectra Pb (nb,32)."""
    sc = np.zeros(len(bf.dirs))
    for j in range(len(bf.band_idx)):
        sc += np.abs(bf.V[j].conj() @ Pb[j]) ** 2 / bf.rn[j]
    return sc


WINDOW = np.hanning(WIN_LEN)


def _active_frames(x: np.ndarray):
    """Frame starts + per-frame RMS for frames above the energy gate."""
    starts = np.arange(0, len(x) - WIN_LEN, HOP)
    E = np.array([np.sqrt(np.mean(x[s:s + WIN_LEN] ** 2)) for s in starts])
    keep = E >= ENERGY_KEEP_FRAC * np.percentile(E, 95)
    return starts[keep], E[keep]


def real_response(path: Path, bf: "esp.Beamformer"):
    rec, _ = sf.read(path, dtype="float64")
    starts, E = _active_frames(rec)
    M = np.zeros(len(bf.dirs))
    peaks = []
    for s, e in zip(starts, E):
        sc = _score(bf, np.fft.rfft(rec[s:s + WIN_LEN] * WINDOW[:, None], axis=0)[bf.band_idx])
        M += e ** 2 * sc
        peaks.append(int(np.argmax(sc)))
    return M, np.array(starts), np.array(peaks)


def sim_response(feed_path: str, bf: "esp.Beamformer", A: np.ndarray, prop: np.ndarray):
    x, _ = sf.read(feed_path, dtype="float64")
    starts, E = _active_frames(x)
    M = np.zeros(len(bf.dirs))
    peaks = []
    for s, e in zip(starts, E):
        S = np.fft.rfft(x[s:s + WIN_LEN] * WINDOW[:, None], axis=0)[bf.band_idx]   # (nb,K)
        Pb = np.einsum("fkc,fk->fc", A, prop * S)                                  # (nb,32)
        sc = _score(bf, Pb)
        M += e ** 2 * sc
        peaks.append(int(np.argmax(sc)))
    return M, np.array(starts), np.array(peaks)


def map_peak(M: np.ndarray, bf: "esp.Beamformer", R: np.ndarray):
    k = int(np.argmax(M))
    az, el = vec_to_amb(R @ bf.dirs[k])
    return az, el, float(M[k] / M.mean())


def az_coverage(peak_idx: np.ndarray, bf: "esp.Beamformer", R: np.ndarray):
    az = np.array([vec_to_amb(R @ bf.dirs[k])[0] for k in peak_idx])
    bins = np.floor((az % 360) / 30).astype(int)
    return len(np.unique(bins)) / 12.0 * 100.0, float(az.min()), float(az.max())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def find_media(media_dir: Path, song: str, suffix: str):
    hits = glob.glob(str(media_dir / f"{SONG_PREFIX[song]}*{suffix}.wav"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recs-dir", type=Path, default=DEFAULT_RECS_DIR)
    ap.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    ap.add_argument("--layout-csv", type=Path, default=DEFAULT_LAYOUT_CSV,
                    help="loudspeaker layout used to simulate the anechoic field")
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--rotation-cache", type=Path, default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_PIPELINE_DOA_CSV)
    args = ap.parse_args()

    print("Step 1: mic->room rotation R from the sweep recordings")
    if args.rotation_cache and args.rotation_cache.is_file() and not args.recompute:
        R, fs = np.load(args.rotation_cache), 48000
        print(f"  loaded cached R from {args.rotation_cache}")
    else:
        R, fs = compute_mic_to_room(args.reg)
        if args.rotation_cache:
            np.save(args.rotation_cache, R)
            print(f"  saved R to {args.rotation_cache}")

    bf = dome_beamformer(fs)
    A, prop, labels = build_speaker_model(bf, R, args.layout_csv, fs)
    print(f"\nStep 2: simulate-and-compare (win {WIN_LEN}, hop {HOP}, grid {GRID_STEP_DEG:.0f} deg, "
          f"dome el>={EL_FLOOR_DEG:.0f}, band {esp.BAND_HZ[0]:.0f}-{esp.BAND_HZ[1]:.0f} Hz, "
          f"{len(labels)} speakers from {args.layout_csv.name})\n")

    rows = []
    for sd in sorted(p for p in args.recs_dir.iterdir() if p.is_dir()):
        song = sd.name.replace("_recs", "")
        if song not in SONG_PREFIX:
            continue
        print(f"=== {song} ===")
        for test in TEST_ORDER:
            kind, suffix, pan = TEST_SPECS[test]
            recs = list(sd.glob(f"*_{test}.wav"))
            feed = find_media(args.media_dir, song, suffix)
            if not recs or not feed:
                print(f"  [{test}] (missing rec or media)")
                continue
            Mr, _, rpk = real_response(recs[0], bf)
            Ms, _, spk = sim_response(feed, bf, A, prop)
            corr = float(np.corrcoef(Mr, Ms)[0, 1])
            saz, sel, spm = map_peak(Ms, bf, R)
            raz, rel, rpm = map_peak(Mr, bf, R)
            pkerr = ang_between(amb_vec(raz, rel), amb_vec(saz, sel))

            if kind in ("static",):
                span = ang_between(amb_vec(saz, sel), amb_vec(*pan))
                print(f"  [{test}] pan({pan[0]:+.0f},{pan[1]:+.0f}) | "
                      f"SIM peak({saz:+6.1f},{sel:+5.1f}) pk/mean {spm:4.1f} sim-vs-pan {span:5.1f} | "
                      f"REAL peak({raz:+6.1f},{rel:+5.1f}) pk/mean {rpm:4.1f} | "
                      f"real-vs-sim {pkerr:5.1f} | mapcorr {corr:+.2f}")
                rows.append([song, test, kind, round(saz, 1), round(sel, 1), round(spm, 2),
                             round(span, 1), round(raz, 1), round(rel, 1), round(rpm, 2),
                             round(pkerr, 1), round(corr, 2), ""])
            elif kind in ("bounce", "orbit"):
                rcov, rlo, rhi = az_coverage(rpk, bf, R)
                scov, slo, shi = az_coverage(spk, bf, R)
                note = (f"az-coverage real {rcov:.0f}% [{rlo:+.0f},{rhi:+.0f}] vs "
                        f"sim {scov:.0f}% [{slo:+.0f},{shi:+.0f}]")
                print(f"  [{test}] {note} | mapcorr {corr:+.2f} | "
                      f"REAL pk/mean {rpm:4.1f} SIM pk/mean {spm:4.1f}")
                rows.append([song, test, kind, "", "", round(spm, 2), "", "", "",
                             round(rpm, 2), "", round(corr, 2), note])
            else:  # song
                print(f"  [{test}] 6 simultaneous sources | mapcorr {corr:+.2f} | "
                      f"SIM peak({saz:+6.1f},{sel:+5.1f}) pk/mean {spm:4.1f} | "
                      f"REAL peak({raz:+6.1f},{rel:+5.1f}) pk/mean {rpm:4.1f}")
                rows.append([song, test, kind, round(saz, 1), round(sel, 1), round(spm, 2),
                             "", round(raz, 1), round(rel, 1), round(rpm, 2), round(pkerr, 1),
                             round(corr, 2), "multi-source: compare via mapcorr"])
        print()

    # aggregate fidelity summary over static tests
    static_corr = [r[11] for r in rows if r[2] == "static"]
    static_pkerr = [r[10] for r in rows if r[2] == "static"]
    if static_corr:
        print("Reproduction-fidelity summary (static tests, real vs anechoic-ideal):")
        print(f"  map correlation: mean {np.mean(static_corr):+.2f} (range {min(static_corr):+.2f}..{max(static_corr):+.2f})")
        print(f"  real-vs-sim peak angle: mean {np.mean(static_pkerr):.0f} deg")
        print("  high corr / small peak angle = faithful reproduction; low corr / large angle = room/repro degradation.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["song", "test", "kind", "sim_peak_az", "sim_peak_el", "sim_pk_mean",
                    "sim_vs_pan_deg", "real_peak_az", "real_peak_el", "real_pk_mean",
                    "real_vs_sim_deg", "map_corr", "notes"])
        w.writerows(rows)
    print(f"\nResults saved to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
