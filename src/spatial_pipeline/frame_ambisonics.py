# Frame-based HOA processing.

import numpy as np

from .frame_processing import (
    split_frames,
    overlap_add,
)

from .ambisonics.encoding.hoa import (
    encode_mono_to_hoa,
)

from .ambisonics.core.conventions import (
    SphericalPosition,
)


def process_hoa_frames(
    signal: np.ndarray,
    positions: list[SphericalPosition],
    order: int,
    frame_size: int,
    hop_size: int,
    normalization: str = "sn3d",
) -> np.ndarray:
    """
    Encode a mono signal into HOA using frame-based processing.[file:2][file:5]
    """

    # Convert input to a consistent audio dtype.
    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    # Validate that the input signal is mono 
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")


    # Split the signal into overlapping frames.
    frames = split_frames(
        signal,
        frame_size,
        hop_size,
    )

    # The trajectory must provide exactly one position per frame —
    # if they don't match, the caller computed n_frames incorrectly,
    # likely using a different frame_size or hop_size than we use here
    if len(positions) != len(frames):
        raise ValueError(
            f"Trajectory length ({len(positions)}) does not match "
            f"number of frames ({len(frames)}). "
            "Recompute the trajectory using the same frame_size and hop_size."
        )


    # Store the encoded HOA frame sequence.
    encoded_frames = []

    for frame, position in zip(frames, positions):

        # Encode the current frame at the given source position.
        encoded = encode_mono_to_hoa(
            frame,
            position,
            order=order,
            normalization=normalization,
        )

        encoded_frames.append(encoded)

    # Return an empty array if no frames were generated.
    if len(encoded_frames) == 0:
        return np.empty((0, 0), dtype=np.float32)

    # Number of HOA channels produced by the encoder.
    n_channels = encoded_frames[0].shape[1]

    # Reconstruct each HOA channel independently.
    reconstructed_channels = []

    for ch in range(n_channels):

        # Collect frames for the current HOA channel.
        channel_frames = [
            frame[:, ch]
            for frame in encoded_frames
        ]

        # Rebuild the channel with overlap-add.
        reconstructed = overlap_add(
            channel_frames,
            frame_size,
            hop_size,
            original_length=len(signal),
        )

        reconstructed_channels.append(reconstructed)

    # Return the reconstructed multichannel HOA signal.
    return np.stack(
        reconstructed_channels,
        axis=1,
    ).astype(np.float32)