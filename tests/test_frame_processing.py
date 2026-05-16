import numpy as np
import pytest
from numpy.testing import assert_allclose
from spatial_pipeline.frame_processing import split_frames, overlap_add


def test_split_frames_returns_expected_frames():
    signal = np.array([0, 1, 2, 3, 4, 5])
    frame_size = 4
    hop_size = 2
    win = np.hanning(frame_size + 2)[1:-1]

    frames = split_frames(signal, frame_size, hop_size)

    expected_frames = [
        np.array([0, 1, 2, 3]) * win,
        np.array([2, 3, 4, 5]) * win,
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
    # Hanning window, we slice away the two zero endpoints, giving a COLA-compliant window
    win = np.hanning(frame_size + 2)[1:-1]

    output = overlap_add(frames, frame_size, hop_size)
    n_frames = len(frames)
    out_len = hop_size * (n_frames - 1) + frame_size
    raw_sum = np.zeros(out_len)
    envelope = np.zeros(out_len)

    for i, f in enumerate(frames):
        s = i * hop_size
        raw_sum[s:s + frame_size] += f
        envelope[s:s + frame_size] += win

    expected_output = np.where(envelope > 1e-8, raw_sum / envelope, raw_sum)
    assert_allclose(output, expected_output)


def test_split_and_overlap_add_pipeline():
    signal = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    frame_size = 4
    hop_size = 2

    frames = split_frames(signal, frame_size, hop_size)
    reconstructed = overlap_add(frames, frame_size, hop_size, original_length=len(signal))

    assert_allclose(reconstructed, signal, atol=1e-6)

def test_perfect_reconstruction_50_percent_overlap():
    signal = np.random.randn(4096)
    frame_size = 512
    hop_size = 256

    frames = split_frames(signal, frame_size, hop_size)
    reconstructed = overlap_add(frames, frame_size, hop_size, original_length=len(signal))
    assert_allclose(reconstructed, signal, atol=1e-6)


def test_perfect_reconstruction_75_percent_overlap():
    signal = np.random.randn(4096)
    frame_size = 512
    hop_size = 128
    frames = split_frames(signal, frame_size, hop_size)
    reconstructed = overlap_add(frames, frame_size, hop_size,
                                original_length=len(signal))
    assert_allclose(reconstructed, signal, atol=1e-6)


def test_perfect_reconstruction_silent_signal():
    signal = np.zeros(1024, dtype=float)
    frame_size = 256
    hop_size = 128
    frames = split_frames(signal, frame_size, hop_size)
    reconstructed = overlap_add(frames, frame_size, hop_size,
                                original_length=len(signal))
    assert_allclose(reconstructed, signal, atol=1e-10)


def test_pad_tail_captures_last_samples():
    signal = np.ones(7, dtype=float)
    frames_no_pad = split_frames(signal, frame_size=4, hop_size=2, pad_tail=False)
    frames_padded = split_frames(signal, frame_size=4, hop_size=2, pad_tail=True)
    assert len(frames_no_pad) == 2
    assert len(frames_padded) == 3


def test_split_frames_rejects_2d_input():
    signal_2d = np.ones((100, 2), dtype=float)
    with pytest.raises(ValueError, match="1D"):
        split_frames(signal_2d, frame_size=32, hop_size=16)
