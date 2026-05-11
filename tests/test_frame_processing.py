import numpy as np
from numpy.testing import assert_allclose
from spatial_pipeline.frame_processing import split_frames, overlap_add


def test_split_frames_returns_expected_frames():
    signal = np.array([0, 1, 2, 3, 4, 5])
    frame_size = 4
    hop_size = 2

    frames = split_frames(signal, frame_size, hop_size)

    expected_frames = [
        np.array([0, 1, 2, 3]),
        np.array([2, 3, 4, 5]),
    ]

    assert len(frames) == len(expected_frames)
    for frame, expected in zip(frames, expected_frames):
        assert_allclose(frame, expected)


def test_overlap_add_reconstructs_expected_signal():
    frames = [
        np.array([0, 1, 2, 3], dtype=float),
        np.array([2, 3, 4, 5], dtype=float),
    ]
    frame_size = 4
    hop_size = 2

    output = overlap_add(frames, frame_size, hop_size)

    expected_output = np.array([0, 1, 4, 6, 4, 5], dtype=float)
    assert_allclose(output, expected_output)


def test_split_and_overlap_add_pipeline():
    signal = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    frame_size = 4
    hop_size = 2

    frames = split_frames(signal, frame_size, hop_size)
    reconstructed = overlap_add(frames, frame_size, hop_size)

    expected = np.array([1, 2, 6, 8, 5, 6], dtype=float)
    assert_allclose(reconstructed, expected)