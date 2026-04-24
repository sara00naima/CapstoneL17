import numpy as np

from spatial_pipeline.ambisonics.encoding.foa import encode_mono_to_foa

ATOL = 1e-6


def _mean_channels(foa):
    W, Y, Z, X = foa.T
    return W.mean(), X.mean(), Y.mean(), Z.mean()


def _assert_direction(
    azimuth_deg,
    elevation_deg,
    expected_wxyz,
    n_samples=32,
):
    s = np.ones(n_samples, dtype=np.float32)

    foa = encode_mono_to_foa(
        s,
        azimuth_rad=np.deg2rad(azimuth_deg),
        elevation_rad=np.deg2rad(elevation_deg),
        convention="basic",
    )

    w, x, y, z = _mean_channels(foa)
    ew, ex, ey, ez = expected_wxyz

    assert np.allclose(w, ew, atol=ATOL), f"W: got {w}, expected {ew}"
    assert np.allclose(x, ex, atol=ATOL), f"X: got {x}, expected {ex}"
    assert np.allclose(y, ey, atol=ATOL), f"Y: got {y}, expected {ey}"
    assert np.allclose(z, ez, atol=ATOL), f"Z: got {z}, expected {ez}"


def test_front():
    _assert_direction(
        azimuth_deg=0.0,
        elevation_deg=0.0,
        expected_wxyz=(1.0, 1.0, 0.0, 0.0),
    )


def test_left():
    _assert_direction(
        azimuth_deg=90.0,
        elevation_deg=0.0,
        expected_wxyz=(1.0, 0.0, 1.0, 0.0),
    )


def test_right():
    _assert_direction(
        azimuth_deg=-90.0,
        elevation_deg=0.0,
        expected_wxyz=(1.0, 0.0, -1.0, 0.0),
    )


def test_back():
    _assert_direction(
        azimuth_deg=180.0,
        elevation_deg=0.0,
        expected_wxyz=(1.0, -1.0, 0.0, 0.0),
    )


def test_up():
    _assert_direction(
        azimuth_deg=0.0,
        elevation_deg=90.0,
        expected_wxyz=(1.0, 0.0, 0.0, 1.0),
    )