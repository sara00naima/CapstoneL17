import numpy as np

from spatial_pipeline.ambisonics.encoding.foa import encode_mono_to_foa
from spatial_pipeline.ambisonics.encoding.hoa import encode_mono_to_hoa

ATOL = 1e-6


def test_foa_basic_equals_hoa_order1_sn3d():
    s = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    azi = np.deg2rad(25.0)
    ele = np.deg2rad(-10.0)

    foa = encode_mono_to_foa(
        s,
        azimuth_rad=azi,
        elevation_rad=ele,
        convention="basic",
    )

    hoa = encode_mono_to_hoa(
        s,
        azimuth_rad=azi,
        elevation_rad=ele,
        order=1,
        normalization="sn3d",
    )

    assert foa.shape == (32, 4)
    assert hoa.shape == (32, 4)
    assert np.allclose(foa, hoa, atol=ATOL)


def test_foa_n3d_like_equals_hoa_order1_n3d():
    s = np.linspace(-0.25, 0.75, 48, dtype=np.float32)
    azi = np.deg2rad(-70.0)
    ele = np.deg2rad(40.0)

    foa = encode_mono_to_foa(
        s,
        azimuth_rad=azi,
        elevation_rad=ele,
        convention="n3d_like",
    )

    hoa = encode_mono_to_hoa(
        s,
        azimuth_rad=azi,
        elevation_rad=ele,
        order=1,
        normalization="n3d",
    )

    assert foa.shape == (48, 4)
    assert hoa.shape == (48, 4)
    assert np.allclose(foa, hoa, atol=ATOL)


def test_hoa_order1_has_four_channels():
    s = np.ones(10, dtype=np.float32)

    hoa = encode_mono_to_hoa(
        s,
        azimuth_rad=0.0,
        elevation_rad=0.0,
        order=1,
        normalization="sn3d",
    )

    assert hoa.shape == (10, 4)