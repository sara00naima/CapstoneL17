from pathlib import Path

import numpy as np

from spatial_pipeline.speaker_layout import (
    Speaker,
    load_speaker_layout,
    layout_labels,
    layout_to_numpy,
    wrap_azimuth_deg,
)

ATOL = 1e-6


def _csv_path() -> Path:
    return Path("measurements_transcription.csv")


def test_wrap_azimuth_deg():
    assert wrap_azimuth_deg(70.0) == 70.0
    assert wrap_azimuth_deg(180.0) == 180.0
    assert wrap_azimuth_deg(190.0) == -170.0
    assert wrap_azimuth_deg(278.0) == -82.0
    assert wrap_azimuth_deg(325.0) == -35.0
    assert wrap_azimuth_deg(340.0) == -20.0


def test_speaker_cartesian_front():
    s = Speaker(
        label="A_test",
        radius_m=2.0,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        cardinal="N",
    )
    expected = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    assert np.allclose(s.cartesian, expected, atol=ATOL)


def test_speaker_cartesian_left():
    s = Speaker(
        label="A_test",
        radius_m=2.0,
        azimuth_deg=90.0,
        elevation_deg=0.0,
        cardinal="E",
    )
    expected = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    assert np.allclose(s.cartesian, expected, atol=ATOL)


def test_speaker_cartesian_up():
    s = Speaker(
        label="A_test",
        radius_m=2.0,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        cardinal="",
    )
    expected = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    assert np.allclose(s.cartesian, expected, atol=ATOL)


def test_load_speaker_layout_count_and_labels():
    speakers = load_speaker_layout(_csv_path())
    assert len(speakers) == 17
    assert layout_labels(speakers)[0] == "A3"
    assert layout_labels(speakers)[-1] == "A19"


def test_cardinal_column_is_read():
    speakers = load_speaker_layout(_csv_path())
    by_label = {s.label: s for s in speakers}

    assert by_label["A4"].cardinal == "N"
    assert by_label["A6"].cardinal == "O"
    assert by_label["A10"].cardinal == "SE"
    assert by_label["A19"].cardinal == ""


def test_known_rows_are_parsed_correctly():
    speakers = load_speaker_layout(_csv_path())
    by_label = {s.label: s for s in speakers}

    a4 = by_label["A4"]
    assert np.isclose(a4.radius_m, 1.966, atol=ATOL)
    assert np.isclose(a4.azimuth_deg, 9.0, atol=ATOL)
    assert np.isclose(a4.elevation_deg, 18.08, atol=ATOL)

    a6 = by_label["A6"]
    assert np.isclose(a6.azimuth_deg, -82.0, atol=ATOL)
    assert np.isclose(a6.elevation_deg, 16.9, atol=ATOL)

    a19 = by_label["A19"]
    assert np.isclose(a19.azimuth_deg, 67.0, atol=ATOL)
    assert np.isclose(a19.elevation_deg, 80.0, atol=ATOL)


def test_layout_to_numpy_shapes():
    speakers = load_speaker_layout(_csv_path())
    azimuth_rad, elevation_rad, cartesian = layout_to_numpy(speakers)

    assert azimuth_rad.shape == (17,)
    assert elevation_rad.shape == (17,)
    assert cartesian.shape == (17, 3)


def test_cartesian_norm_matches_radius():
    speakers = load_speaker_layout(_csv_path())
    for s in speakers:
        norm = np.linalg.norm(s.cartesian)
        assert np.isclose(norm, s.radius_m, atol=1e-5), (
            f"{s.label}: norm={norm}, radius={s.radius_m}"
        )