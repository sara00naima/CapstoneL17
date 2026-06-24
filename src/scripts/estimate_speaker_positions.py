#!/usr/bin/env python3
"""
estimate_speaker_positions.py
=============================
Estimate the 3D positions (Distance, Azimuth, Elevation) of the 17 Cremona
museum loudspeakers from Eigenmike EM32 sweep recordings (32 ch, 48 kHz).

Each ``A*.wav`` recording contains the log-sine sweep played by ONE loudspeaker;
the reference excitation is ``sweep_48000_50_22000.wav`` (mono, 10 s, 50-22 kHz).

Pipeline
--------
1. Wiener spectral deconvolution of the reference sweep -> 32-channel RIR
   (inverse filter  conj(S) / (|S|^2 + reg * max|S|^2) ).  The direct sound is
   located as the global peak of the broadband RMS envelope over the FULL
   (circular) deconvolution buffer -- some takes start the sweep so early that
   the impulse wraps to the end of the buffer, so a plain [0:N] search misses it.
2. A short window at the direct peak -> 32-ch snapshot of the direct wavefront.
3. Direction of arrival via RIGID-SPHERE plane-wave-decomposition beamforming
   (the proper EM32 model: spherical-harmonic order 4 with rigid-sphere modal
   coefficients b_n(ka)).  A naive 1st-order pseudo-intensity instead badly
   distorts elevation on this array.
4. The 17 DOAs (mic frame) are aligned to the room frame with a single rigid
   (Kabsch) rotation, calibrated against ``measurements_transcription.csv``.
   IMPORTANT: the reference directions are taken AS SEEN FROM THE MIC -- the
   Eigenmike sits 0.61 m above the CSV reference point, so each source appears
   at a different (parallax-shifted) elevation than the floor-referenced CSV
   value.  An iterative outlier rejection drops any speaker that still does not
   fit the rigid rotation (e.g. residual floor-reflection bias on the lowest
   ring), so the orientation is fixed by the cleanly localised sources.
5. Each source is placed at its (copied) distance along the estimated mic ray
   and re-expressed from the reference point -> (Distance, Azimuth, Elevation).
   Distance is COPIED from the reference CSV: the takes are not time-synchronised
   to the playback (the direct-path delays span seconds), so time-of-flight
   cannot recover absolute distance from these files.
6. Output CSV identical in format to ``measurements_transcription_original.csv``
   (azimuth in the room compass frame = transcription azimuth + 70 deg).

Usage
-----
    python src/scripts/estimate_speaker_positions.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import spherical_jn, spherical_yn, eval_legendre
from scipy.signal.windows import hann

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
C_SOUND = 343.0          # speed of sound [m/s]
MIC_HEIGHT_M = 0.61      # Eigenmike height ABOVE the CSV reference point [m]
A_RADIUS = 0.042         # EM32 rigid-sphere radius [m]
AZ_TRANS_TO_ORIGINAL = 70.0   # original azimuth = transcription azimuth + 70 deg

# DOA analysis settings (validated against measurements_transcription.csv).
# The window is deliberately SHORT (~1 ms): the lowest ring (A15-A18) sits close
# to the floor, so a strong early reflection arrives only ~1 ms after the direct
# sound; a longer window blends them and drags the estimated elevation ~30 deg
# too low.  A 48-sample gate keeps mostly the direct wavefront.  The high band
# (up to ~5 kHz, the EM32 order-4 spatial-aliasing limit) keeps enough frequency
# bins for the modal beamformer despite the short window.
SH_ORDER = 4
BAND_HZ = (400.0, 5000.0)
WIN_LEN = 48            # direct-sound snapshot length [samples] (~1 ms @ 48 kHz)
WIN_PRE = 8             # samples kept before the direct peak
GRID_STEP_DEG = 3.0     # DOA search-grid resolution
CALIB_OUTLIER_DEG = 18.0  # speakers above this residual are dropped from the fit

# EM32 capsule directions (mh acoustics standard layout).
# Azimuth CCW-from-front (+90 = left); elevation = 90 - colatitude.
EM32_AZ_DEG = np.array(
    [0, 32, 0, 328, 0, 45, 69, 45, 0, 315, 291, 315, 91, 90, 90, 89,
     180, 212, 180, 148, 180, 225, 249, 225, 180, 135, 111, 135, 269, 270, 270, 271],
    dtype=float,
)
EM32_COLAT_DEG = np.array(
    [69, 90, 111, 90, 32, 55, 90, 125, 148, 125, 90, 55, 21, 58, 121, 159,
     69, 90, 111, 90, 32, 55, 90, 125, 148, 125, 90, 55, 21, 58, 122, 159],
    dtype=float,
)
EM32_EL_DEG = 90.0 - EM32_COLAT_DEG

CARDINALS_IT = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]  # O = Ovest (West)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REC_DIR = PROJECT_ROOT / "sweeps-40-00.wav"
DEFAULT_SWEEP = DEFAULT_REC_DIR / "sweep_48000_50_22000.wav"
DEFAULT_TRANS_CSV = PROJECT_ROOT / "measurements_transcription.csv"
DEFAULT_OUT_CSV = PROJECT_ROOT / "estimated_positions.csv"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def capsule_unit_vectors() -> np.ndarray:
    """(32, 3) capsule directions in the mic frame: x=front, y=left, z=up."""
    az = np.deg2rad(EM32_AZ_DEG)
    el = np.deg2rad(EM32_EL_DEG)
    return np.column_stack([np.cos(el) * np.cos(az),
                            np.cos(el) * np.sin(az),
                            np.sin(el)])


UQ = capsule_unit_vectors()


def grid_directions(step_deg: float):
    els = np.arange(-90.0, 90.0 + 1e-6, step_deg)
    azs = np.arange(0.0, 360.0, step_deg)
    A, E = np.meshgrid(azs, els)
    a, e = np.deg2rad(A.ravel()), np.deg2rad(E.ravel())
    dirs = np.column_stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return dirs


def compass_to_room_vec(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Compass az/el -> room Cartesian (x=North/front, y=West, z=up). CW (East=-y)."""
    a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.column_stack([np.cos(e) * np.cos(a), -np.cos(e) * np.sin(a), np.sin(e)])


def room_vec_to_compass(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    az = np.rad2deg(np.arctan2(-v[1], v[0])) % 360.0
    el = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return az, el


def cardinal_label(az_deg: float) -> str:
    return CARDINALS_IT[int(round((az_deg % 360.0) / 45.0)) % 8]


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Proper rotation R (det=+1) with Q ~ R @ P (rows are vectors)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# --------------------------------------------------------------------------- #
# Rigid-sphere modal beamformer
# --------------------------------------------------------------------------- #
def _bn(n: int, ka: np.ndarray) -> np.ndarray:
    """Rigid-sphere modal coefficient b_n(ka) (global constant omitted)."""
    jn = spherical_jn(n, ka); jnp = spherical_jn(n, ka, derivative=True)
    yn = spherical_yn(n, ka); ynp = spherical_yn(n, ka, derivative=True)
    hn = jn - 1j * yn; hnp = jnp - 1j * ynp           # spherical Hankel 2nd kind
    return (1j ** n) * (jn - (jnp / hnp) * hn)


class Beamformer:
    """Bartlett plane-wave-decomposition beamformer for the rigid EM32 sphere."""

    def __init__(self, dirs: np.ndarray, order: int, band: tuple[float, float],
                 win_len: int, fs: int):
        self.dirs = dirs
        freqs = np.fft.rfftfreq(win_len, 1.0 / fs)
        self.band_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        cosg = dirs @ UQ.T                                       # (G, 32)
        Pn = np.stack([(2 * n + 1) * eval_legendre(n, cosg) for n in range(order + 1)])
        self.V, self.rn = [], []
        for b in self.band_idx:
            ka = 2 * np.pi * freqs[b] * A_RADIUS / C_SOUND
            bvec = np.array([_bn(n, ka) for n in range(order + 1)])
            V = np.tensordot(bvec, Pn, axes=(0, 0))             # (G, 32) complex
            self.V.append(V)
            self.rn.append(np.sum(np.abs(V) ** 2, axis=1) + 1e-12)
        self.win = hann(win_len)

    def doa(self, snapshot: np.ndarray) -> np.ndarray:
        """32-ch direct-sound snapshot -> mic-frame DOA unit vector."""
        P = np.fft.rfft(snapshot * self.win[:, None], axis=0)
        score = np.zeros(len(self.dirs))
        for j, b in enumerate(self.band_idx):
            score += (np.abs(self.V[j].conj() @ P[b]) ** 2) / self.rn[j]
        return self.dirs[int(np.argmax(score))]


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def inverse_filter(sweep: np.ndarray, n_fft: int, reg: float) -> np.ndarray:
    S = np.fft.rfft(sweep, n=n_fft)
    mag2 = np.abs(S) ** 2
    return np.conj(S) / (mag2 + reg * mag2.max())


def direct_snapshot(rec: np.ndarray, inv: np.ndarray, n_fft: int):
    """Deconvolve all channels, find the direct peak (circular), window it."""
    rir = np.fft.irfft(np.fft.rfft(rec, n=n_fft, axis=0) * inv[:, None], n=n_fft, axis=0)
    env = np.sqrt(np.mean(rir ** 2, axis=1))
    p0 = int(np.argmax(env))                                   # full circular buffer
    idx = (np.arange(-WIN_PRE, WIN_LEN - WIN_PRE) + p0) % n_fft
    return rir[idx], p0, float(env[p0])


# --------------------------------------------------------------------------- #
# Reference CSV
# --------------------------------------------------------------------------- #
def load_reference(csv_path: Path) -> dict[str, dict]:
    ref: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)
        for row in reader:
            if len(row) < 5 or not row[0].strip().startswith("A"):
                continue
            ref[row[0].strip().upper()] = {
                "distance": float(row[1].replace(",", ".")),
                "az": float(row[2].replace(",", ".")),
                "el": float(row[4].replace(",", ".")),
            }
    return ref


def label_from_filename(path: Path) -> str | None:
    m = re.match(r"[Aa](\d+)$", path.stem)
    return f"A{int(m.group(1))}" if m else None


def sort_key(label: str) -> int:
    return int(label[1:])


# --------------------------------------------------------------------------- #
# Reference-point re-projection (mic is MIC_HEIGHT_M above the CSV reference)
# --------------------------------------------------------------------------- #
def mic_pos() -> np.ndarray:
    return np.array([0.0, 0.0, MIC_HEIGHT_M])


def reference_dir_from_mic(r: dict) -> np.ndarray:
    """CSV (distance, compass az, el from reference point) -> unit direction from the mic."""
    pos = compass_to_room_vec(np.array([r["az"]]), np.array([r["el"]]))[0] * r["distance"]
    v = pos - mic_pos()
    return v / np.linalg.norm(v)


def mic_distance_from_reference(distance_ref: float, doa_room: np.ndarray) -> float:
    """Intersect the mic ray with the sphere of radius distance_ref about the reference."""
    m = mic_pos()
    b = 2.0 * float(m @ doa_room)
    c = float(m @ m) - distance_ref ** 2
    return (-b + np.sqrt(max(0.0, b * b - 4.0 * c))) / 2.0


def project_to_reference(doa_room: np.ndarray, r_mic: float):
    """Source at direction doa_room and distance r_mic from the mic -> (dist, az, el) from reference."""
    P = mic_pos() + r_mic * doa_room
    dist = float(np.linalg.norm(P))
    az, el = room_vec_to_compass(P)
    return dist, az, el


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(doa_mic: np.ndarray, ref_dirs: np.ndarray):
    """Rigid mic->room rotation with iterative rejection of floor-corrupted speakers."""
    mask = np.ones(len(doa_mic), bool)
    R = kabsch(doa_mic, ref_dirs)
    for _ in range(8):
        R = kabsch(doa_mic[mask], ref_dirs[mask])
        res = np.rad2deg(np.arccos(np.clip(np.sum((doa_mic @ R.T) * ref_dirs, axis=1), -1, 1)))
        new = res < CALIB_OUTLIER_DEG
        if new.sum() < 6 or (new == mask).all():
            break
        mask = new
    return R, res, mask


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Estimate museum loudspeaker positions from EM32 sweep recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--recordings-dir", type=Path, default=DEFAULT_REC_DIR)
    ap.add_argument("--sweep-file", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--reference-csv", type=Path, default=DEFAULT_TRANS_CSV)
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--reg", type=float, default=1e-4, help="Wiener inverse-filter regularisation")
    args = ap.parse_args()

    if not args.recordings_dir.is_dir():
        print(f"ERROR: recordings folder not found:\n  {args.recordings_dir}", file=sys.stderr)
        return 1
    if not args.sweep_file.is_file():
        print(f"ERROR: sweep file not found:\n  {args.sweep_file}", file=sys.stderr)
        return 1

    sweep, fs = sf.read(args.sweep_file, dtype="float64")
    if sweep.ndim > 1:
        sweep = sweep[:, 0]
    print(f"Sweep: {len(sweep)} samples | {fs} Hz | {len(sweep)/fs:.1f}s")

    ref = load_reference(args.reference_csv)
    bf = Beamformer(grid_directions(GRID_STEP_DEG), SH_ORDER, BAND_HZ, WIN_LEN, fs)

    wavs = sorted((p for p in args.recordings_dir.glob("*.wav") if label_from_filename(p)),
                  key=lambda p: sort_key(label_from_filename(p)))
    if not wavs:
        print(f"ERROR: no A*.wav files in:\n  {args.recordings_dir}", file=sys.stderr)
        return 1
    print(f"Files: {len(wavs)} -> {[label_from_filename(p) for p in wavs]}\n")

    results = []
    for p in wavs:
        label = label_from_filename(p)
        rec, fsr = sf.read(p, dtype="float64")
        if rec.ndim != 2 or rec.shape[1] != 32:
            print(f"  [ERROR] {label}: expected 32 channels, got {rec.shape}", file=sys.stderr)
            continue
        n_fft = len(rec) + len(sweep)
        inv = inverse_filter(sweep, n_fft, args.reg)
        snap, p0, peak = direct_snapshot(rec, inv, n_fft)
        doa_mic = bf.doa(snap)
        results.append({"label": label, "doa_mic": doa_mic, "p0": p0, "peak": peak})

    # ---- calibrate mic -> room (against mic-view reference directions) ----
    common = [r for r in results if r["label"] in ref]
    D = np.array([r["doa_mic"] for r in common])
    Qmic = np.array([reference_dir_from_mic(ref[r["label"]]) for r in common])
    R, resid, inliers = calibrate(D, Qmic)
    front_az, _ = room_vec_to_compass(R @ np.array([1.0, 0.0, 0.0]))
    excluded = [common[i]["label"] for i in range(len(common)) if not inliers[i]]
    print("Mic->room calibration (Kabsch vs measurements_transcription.csv, mic-view):")
    print(f"  inliers {inliers.sum()}/{len(common)} | residual mean {resid[inliers].mean():.1f} deg "
          f"(all {resid.mean():.1f} deg)")
    print(f"  EM32 front -> transcription azimuth {front_az:.1f} deg")
    if excluded:
        print(f"  excluded from fit (floor-reflection-biased elevation): {excluded}")

    # ---- assemble positions in the original compass frame & write CSV ----
    rows = []
    for r in sorted(results, key=lambda r: sort_key(r["label"])):
        label = r["label"]
        doa_room = R @ r["doa_mic"]
        doa_room /= np.linalg.norm(doa_room)
        dist_ref = ref[label]["distance"] if label in ref else 1.7
        r_mic = mic_distance_from_reference(dist_ref, doa_room)
        dist, az_trans, el = project_to_reference(doa_room, r_mic)
        az_orig = (az_trans + AZ_TRANS_TO_ORIGINAL) % 360.0
        rows.append((label, dist, az_orig, el, label in excluded))

    write_csv(args.output_csv, rows)

    print("\nEstimated positions (original compass frame):")
    print(f"  {'Spk':<5}{'Dist':>8}{'Azim':>7}  {'Card':<4}{'Elev':>8}   note")
    for label, dist, az, el, exc in rows:
        note = "elevation uncertain (floor reflection)" if exc else ""
        print(f"  {label:<5}{dist:>8.3f}{az:>7.0f}  {cardinal_label(az):<4}{el:>8.1f}   {note}")
    print(f"\nDistance is copied from {args.reference_csv.name} "
          f"(takes not time-synchronised -> distance not recoverable from audio).")
    print(f"Results saved to: {args.output_csv}")
    return 0


def write_csv(path: Path, rows) -> None:
    """Write CSV identical in format to measurements_transcription_original.csv."""
    lines = ["Loudspeaker;Distance;Azimuth;;Elevation;setup-height"]
    for i, (label, dist, az, el, _exc) in enumerate(rows):
        height = "0.736" if i == 0 else ""
        lines.append(f"{label};{dist:.3f};{az:.0f}; {cardinal_label(az)};{el:.1f};{height}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
