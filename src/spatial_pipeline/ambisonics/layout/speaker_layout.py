from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.conventions import deg2rad, sph2cart


def _parse_float(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("Empty numeric field")
    return float(value.replace(",", "."))


def wrap_azimuth_deg(angle_deg: float) -> float:
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    if np.isclose(wrapped, -180.0):
        return 180.0
    return float(wrapped)


@dataclass(frozen=True)
class Speaker:
    label: str
    radius_m: float
    azimuth_deg: float
    elevation_deg: float
    cardinal: str = ""

    @property
    def azimuth_rad(self) -> float:
        return float(deg2rad(self.azimuth_deg))

    @property
    def elevation_rad(self) -> float:
        return float(deg2rad(self.elevation_deg))

    @property
    def unit_vector(self) -> np.ndarray:
        return sph2cart(self.azimuth_rad, self.elevation_rad)

    @property
    def cartesian(self) -> np.ndarray:
        return self.radius_m * self.unit_vector


def speaker_from_fields(fields: list[str]) -> Speaker:
    if len(fields) < 5:
        raise ValueError(f"Row has too few columns: {fields}")

    label = fields[0].strip()
    radius_m = _parse_float(fields[1])
    azimuth_deg = wrap_azimuth_deg(_parse_float(fields[2]))
    cardinal = fields[3].strip().upper()
    elevation_deg = _parse_float(fields[4])

    return Speaker(
        label=label,
        radius_m=radius_m,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        cardinal=cardinal,
    )


def load_speaker_layout(csv_path: str | Path) -> list[Speaker]:
    csv_path = Path(csv_path)
    speakers: list[Speaker] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV is empty")

        for fields in reader:
            if not fields or len(fields) < 5:
                continue

            label = fields[0].strip()
            if not label or not label.startswith("A"):
                continue

            speakers.append(speaker_from_fields(fields))

    speakers.sort(key=lambda s: int(s.label[1:]))
    return speakers


def layout_to_numpy(
    speakers: list[Speaker],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_rad = np.array([s.azimuth_rad for s in speakers], dtype=np.float64)
    elevation_rad = np.array([s.elevation_rad for s in speakers], dtype=np.float64)
    cartesian = np.stack([s.cartesian for s in speakers], axis=0)
    return azimuth_rad, elevation_rad, cartesian


def layout_labels(speakers: list[Speaker]) -> list[str]:
    return [s.label for s in speakers]