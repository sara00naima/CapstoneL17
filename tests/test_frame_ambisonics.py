import numpy as np

from spatial_pipeline.frame_ambisonics import (
    process_hoa_frames,
)
from spatial_pipeline.ambisonics.core.conventions import (
    SphericalPosition,
)


def test_process_foa_frames_shape():
    # Use a constant mono input to validate output dimensionality only.
    signal = np.ones(4096)

    # Place the source on the horizontal front direction.
    position = SphericalPosition(
        azimuth=0.0,
        elevation=0.0,
    )

    # Process the signal frame by frame as first-order Ambisonics.
    foa = process_hoa_frames(
        signal,
        position,
        order=1,
        frame_size=1024,
        hop_size=512,
    )

    # FOA output is expected to be a 2D array: samples x channels.
    assert foa.ndim == 2

    # First-order output contains 4 channels.
    assert foa.shape[1] == 4


def test_process_hoa_frames_front_direction():
    # Use a constant signal to inspect the directional channel distribution.
    signal = np.ones(4096)

    # Encode a source located straight ahead.
    position = SphericalPosition(
        azimuth=0.0,
        elevation=0.0,
    )

    hoa = process_hoa_frames(
        signal,
        position,
        order=1,
        frame_size=1024,
        hop_size=512,
    )

    # Split FOA channels according to the expected ACN ordering.
    w = hoa[:, 0]
    y = hoa[:, 1]
    z = hoa[:, 2]
    x = hoa[:, 3]

    # A frontal source must generate a non-zero front-back component.
    assert np.mean(np.abs(x)) > 0.1

    # A centered frontal source should not excite the left-right component.
    assert np.allclose(
        y,
        0.0,
        atol=1e-5,
    )

    # A source on the horizontal plane should not excite the vertical component.
    assert np.allclose(
        z,
        0.0,
        atol=1e-5,
    )