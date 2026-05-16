import numpy as np

def split_frames(
    signal: np.ndarray,
    frame_size: int,
    hop_size: int,
        *,
    pad_tail: bool = True,
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
    pad_tail : bool, optional
        If True (default), zero-pad the signal so the last samples are
        always captured in a complete frame.

    Returns
    -------
    list[np.ndarray]
        List of Hann-windowed frames, each of length frame_size.

    """
    signal = np.asarray(signal, dtype=np.float64)

    # to ensure we have a 1D signal
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")

    if pad_tail:
            # How many samples into the current hop period are we at the end?
            remainder = (len(signal) - frame_size) % hop_size
            if remainder != 0:
                # Zero-pad just enough to complete the final hop, so the last
                # samples of the signal land in a full frame rather than being lost
                pad_length = hop_size - remainder
                signal = np.concatenate([signal, np.zeros(pad_length, dtype=np.float64)])
    
    # Hanning window, we slice away the two zero endpoints, giving a COLA-compliant window
    win = np.hanning(frame_size + 2)[1:-1]

    frames = []

    # Move a fixed-size window across the signal.
    for start in range(0, len(signal) - frame_size + 1, hop_size):
        end = start + frame_size
        frame = signal[start:end] * win
        frames.append(frame)

    return frames


def overlap_add(
    frames: list[np.ndarray],
    frame_size: int,
    hop_size: int,
    *,
    original_length: int | None = None,
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
    original_length : int or None, optional
        If provided, the output is trimmed to exactly this many samples,
        removing any zero-padding that split_frames may have added.


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
        return np.array([], dtype=np.float64)

    n_frames = len(frames)
    output_length = hop_size * (n_frames - 1) + frame_size  # total output length
    
    output = np.zeros(output_length, dtype=np.float64)
    envelope = np.zeros(output_length, dtype=np.float64)

    # Hanning window, we slice away the two zero endpoints, giving a COLA-compliant window
    win = np.hanning(frame_size + 2)[1:-1]

    for i, frame in enumerate(frames):
        start = i * hop_size
        end = start + frame_size

        # Sum the current frame into the output buffer.
        output[start:end]   += np.asarray(frame, dtype=np.float64)
        envelope[start:end] += win

    # Divide by the envelope to recover the original amplitude
    epsilon = 1e-8
    output = np.where(envelope > epsilon, output / envelope, output)

    # Remove any zero-padding that split_frames added, restoring the
    # output to the original signal length
    if original_length is not None:
        output = output[:original_length]

    return output