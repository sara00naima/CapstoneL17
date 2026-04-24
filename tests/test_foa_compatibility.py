import numpy as np

from spatial_pipeline.ambisonics.core.conventions import sph2cart
from spatial_pipeline.ambisonics.encoding.foa import encode_mono_to_foa

ATOL = 1e-6


def _expected_foa_old(signal, azimuth_rad, elevation_rad, convention):
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    x, y, z = sph2cart(azimuth_rad, elevation_rad)

    if convention == "basic":
        gains = np.array([1.0, y, z, x], dtype=np.float32)
    elif convention == "n3d_like":
        g = np.sqrt(3.0)
        gains = np.array([1.0, g * y, g * z, g * x], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    return signal[:, None] * gains[None, :]


def test_foa_basic_matches_old_formula_front():
    s = np.ones(16, dtype=np.float32)
    azi = np.deg2rad(0.0)
    ele = np.deg2rad(0.0)

    got = encode_mono_to_foa(s, azi, ele, convention="basic")
    expected = _expected_foa_old(s, azi, ele, convention="basic")

    assert got.shape == (16, 4)
    assert np.allclose(got, expected, atol=ATOL)


def test_foa_basic_matches_old_formula_generic_direction():
    s = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    azi = np.deg2rad(37.0)
    ele = np.deg2rad(22.0)

    got = encode_mono_to_foa(s, azi, ele, convention="basic")
    expected = _expected_foa_old(s, azi, ele, convention="basic")

    assert got.shape == (32, 4)
    assert np.allclose(got, expected, atol=ATOL)


def test_foa_n3d_like_matches_old_formula_generic_direction():
    s = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    azi = np.deg2rad(-55.0)
    ele = np.deg2rad(30.0)

    got = encode_mono_to_foa(s, azi, ele, convention="n3d_like")
    expected = _expected_foa_old(s, azi, ele, convention="n3d_like")

    assert got.shape == (64, 4)
    assert np.allclose(got, expected, atol=ATOL)


def test_foa_channel_order_is_w_y_z_x():
    s = np.ones(8, dtype=np.float32)
    azi = np.deg2rad(90.0)
    ele = np.deg2rad(0.0)

    foa = encode_mono_to_foa(s, azi, ele, convention="basic")

    w = foa[:, 0]
    y = foa[:, 1]
    z = foa[:, 2]
    x = foa[:, 3]

    assert np.allclose(w, 1.0, atol=ATOL)
    assert np.allclose(y, 1.0, atol=ATOL)
    assert np.allclose(z, 0.0, atol=ATOL)
    assert np.allclose(x, 0.0, atol=ATOL)


def test_foa_rejects_unknown_convention():
    s = np.ones(8, dtype=np.float32)

    try:
        encode_mono_to_foa(
            s,
            azimuth_rad=0.0,
            elevation_rad=0.0,
            convention="wrong",
        )
    except ValueError as exc:
        assert "Unsupported convention" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid convention")