import numpy as np


def split_frames(
    signal: np.ndarray,
    frame_size: int,
    hop_size: int,
) -> list[np.ndarray]:
    """
    Split a 1D signal into overlapping frames.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal as a 1D NumPy array.
    frame_size : int
        Number of samples in each frame.
    hop_size : int
        Number of samples between consecutive frame starts.

    Returns
    -------
    list[np.ndarray]
        List of extracted frames.

    Notes
    -----
    Only complete frames are returned. Incomplete tail samples are ignored.
    """
    signal = np.asarray(signal)
    frames = []

    # Move a fixed-size window across the signal.
    for start in range(0, len(signal) - frame_size + 1, hop_size):
        end = start + frame_size
        frames.append(signal[start:end])

    return frames


def overlap_add(
    frames: list[np.ndarray],
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    """
    Reconstruct a 1D signal from overlapping frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of frames to be summed into a single output signal.
    frame_size : int
        Expected number of samples in each frame.
    hop_size : int
        Number of samples between consecutive frame starts.

    Returns
    -------
    np.ndarray
        Reconstructed output signal.

    Notes
    -----
    Overlapping samples are added together. This is the basic overlap-add
    reconstruction step used in frame-based audio processing.
    """
    if not frames:
        return np.array([])

    n_frames = len(frames)
    output_length = hop_size * (n_frames - 1) + frame_size
    output = np.zeros(output_length)

    for i, frame in enumerate(frames):
        start = i * hop_size
        end = start + frame_size

        # Sum the current frame into the output buffer.
        output[start:end] += frame

    return output