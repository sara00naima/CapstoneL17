"""
export_iem_layout.py
--------------------
Reads the file measurements_transcription.csv e generates museum_17ch_iem.json
ready to be imported in IEM AllRADecoder.

ATTENTION - Angular convention:
  The CSV uses clockwise compass convention (0°=Front, 90°=Right, 270°=Left).
  Both the project and IEM use counter-clockwise ambisonics convention (+90°=Left).
  → CSV azimuth must be negated on load.

Usage:
    python export_iem_layout.py
    python export_iem_layout.py --csv path/al/file.csv --out layout.json
"""

import csv
import json
import argparse
from pathlib import Path


def parse_float(value: str, default: float | None = None) -> float:
    """Accepts both '.' and ',' as decimal separators.

    If the field is empty and ``default`` is given, returns the default
    instead of raising. The DOA estimation script leaves the Distance/Radius
    column blank (it measures direction only), so the radius falls back to
    ``default`` for those rows.
    """
    value = value.strip().replace(",", ".")
    if not value:
        if default is not None:
            return default
        raise ValueError("empty numeric field")
    return float(value)


def wrap_azimuth(angle_deg: float) -> float:
    """Normalizes the azimuth to the range [-180, +180]."""
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    return 180.0 if abs(wrapped + 180.0) < 1e-9 else wrapped


def load_speakers_from_csv(csv_path: Path, default_radius: float = 1.0) -> list[dict]:
    """
    Reads the museum CSV and returns a list of speaker dictionaries.
    Expected CSV format (delimiter ';'):
        label ; radius_m ; azimuth_deg ; cardinal ; elevation_deg
        A1    ; 2.5      ; 0.0         ; N        ; 0.0
    """
    speakers = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV vuoto.")

        for fields in reader:
            if not fields or len(fields) < 5:
                continue
            label = fields[0].strip()
            if not label or not label.startswith("A"):
                continue

            radius_m    = parse_float(fields[1], default=default_radius)
            # Compass CSV (90=Right) -> IEM ambisonics (+90=Left): negate. See
            # module docstring; without this the whole layout is mirrored L<->R.
            azimuth_deg = wrap_azimuth(-parse_float(fields[2]))
            # fields[3] = cardinal (N/S/E/W ecc.) — non serve per IEM
            elevation_deg = parse_float(fields[4])

            speakers.append({
                "label":         label,
                "azimuth_deg":   azimuth_deg,
                "elevation_deg": elevation_deg,
                "radius_m":      radius_m,
            })

    # Ordina per numero di speaker (A1, A2, ... A17)
    speakers.sort(key=lambda s: int(s["label"][1:]))
    return speakers


def build_iem_json(speakers: list[dict], layout_name: str = "Museum_17ch") -> dict:
    """
    Builds the JSON dictionary in the format accepted by IEM AllRADecoder.

    Expected structure from AllRADecoder:
    {
      "LoudspeakerLayout": {
        "Name": "...",
        "Loudspeakers": [
          {
            "Azimuth":     float,   // gradi, +90 = sinistra
            "Elevation":   float,   // gradi, 0 = orizzonte
            "Radius":      float,   // metri (rilevante solo per speaker immaginari)
            "IsImaginary": bool,    // false = speaker reale
            "Channel":     int,     // 1-indexed, corrisponde al canale audio
            "Gain":        float    // 1.0 per speaker reali
          },
          ...
        ]
      }
    }
    """
    loudspeakers = []

    for i, spk in enumerate(speakers):
        loudspeakers.append({
            "Azimuth":     round(spk["azimuth_deg"],   4),
            "Elevation":   round(spk["elevation_deg"], 4),
            "Radius":      round(spk["radius_m"],      4),
            "IsImaginary": False,
            "Channel":     i + 1,
            "Gain":        1.0
        })

    # All real speakers are above the horizon, so the origin falls outside the
    # convex hull — AllRADecoder raises Error 4. An imaginary south-pole speaker
    # closes the hull without driving any physical channel.
    loudspeakers.append({
        "Azimuth":     0.0,
        "Elevation":   -90.0,
        "Radius":      0.01,
        "IsImaginary": True,
        "Channel":     len(speakers) + 1,
        "Gain":        1
    })

    return {
        "LoudspeakerLayout": {
            "Name": layout_name,
            "Loudspeakers": loudspeakers
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Export museum speaker layout → IEM AllRADecoder JSON")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "measurements_transcription.csv",
        help="Path to the measurements_transcription.csv file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "museum_17ch_iem.json",
        help="Path to the output JSON file (default: museum_17ch_iem.json in the same directory)"
    )
    parser.add_argument(
        "--default-radius",
        type=float,
        default=1.0,
        help="Radius [m] used for real speakers when the CSV Distance column "
             "is empty (the DOA estimator measures direction only)"
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    print(f"Reading layout from: {args.csv}")
    speakers = load_speakers_from_csv(args.csv, default_radius=args.default_radius)
    print(f"  Found {len(speakers)} speakers: {[s['label'] for s in speakers]}")

    layout = build_iem_json(speakers)
    args.out.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    print(f"\nJSON exported to: {args.out}")
    print("\nPosition summary:")
    print(f"  {'Label':<6}  {'Ch':>3}  {'Azimuth':>9}  {'Elevation':>10}")
    print(f"  {'-'*6}  {'-'*3}  {'-'*9}  {'-'*10}")
    for i, spk in enumerate(speakers):
        print(f"  {spk['label']:<6}  {i+1:>3}  {spk['azimuth_deg']:>8.1f}°  {spk['elevation_deg']:>9.1f}°")

    print("\nNext step:")
    print("  1. Open IEM AllRADecoder in the DAW")
    print("  2. Click 'Load' or 'Import' -> select museum_17ch_iem.json")
    print("  3. Verify that the blue points in the 3D viewer correspond to the museum layout")
    print("  NOTE: an imaginary south-pole speaker (Ch 18) is included automatically to satisfy the AllRADecoder convex-hull requirement.")


if __name__ == "__main__":
    main()