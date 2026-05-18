import numpy as np
from scipy.signal import fftconvolve
import sofar

from .ambisonics.decoding.decode_to_speakers import calculate_decoder_matrix, decode_hoa_to_speakers
from .ambisonics.core.conventions import deg2rad


def render_binaural(
    ambisonic_audio: np.ndarray, # (samples, channels) HOA bus
    sofa_path: str, # path to a SimpleFreeFieldHRIR SOFA file
    order: int = 3,
    normalization: str = "sn3d",
) -> np.ndarray: # (samples, 2) float32 [left, right]
    """
    Renders binaural stereo from an ambisonic bus using the virtual loudspeaker
    method. Every HRTF measurement in the SOFA file is treated as one virtual
    speaker: the HOA signal is decoded to those directions, each feed is
    convolved with its left/right HRIR, and the results are summed to stereo.
    """

    # Load the HRTF dataset from the SOFA file
    hrtf = sofar.read_sofa(sofa_path)

    # SourcePosition: (M, 3) in degrees [azimuth, elevation, radius]
    # SOFA azimuth is counter-clockwise positive (= left), matching our convention
    azimuth_rad   = deg2rad(hrtf.SourcePosition[:, 0])
    elevation_rad = deg2rad(hrtf.SourcePosition[:, 1])

    # Data_IR: (M, 2, N) — M positions × 2 ears × N IR samples
    hrirs = hrtf.Data_IR

    # a mode-matching decoder that maps the HOA bus to one feed per
    # virtual speaker (each SOFA measurement position is one virtual speaker)
    decoder_matrix = calculate_decoder_matrix(
        azimuth_rad, elevation_rad, order, normalization
    )

    # Decode the ambisonic bus to virtual speaker feeds: (samples, M)
    virtual_feeds = decode_hoa_to_speakers(ambisonic_audio, decoder_matrix)

    n_samples  = ambisonic_audio.shape[0]
    ir_len     = hrirs.shape[2]
    output_len = n_samples + ir_len - 1

    left  = np.zeros(output_len, dtype=np.float64)
    right = np.zeros(output_len, dtype=np.float64)

    # convolve each virtual speaker feed with its left and right HRIR and accumulate
    for i in range(virtual_feeds.shape[1]):
        left  += fftconvolve(virtual_feeds[:, i], hrirs[i, 0, :])
        right += fftconvolve(virtual_feeds[:, i], hrirs[i, 1, :])

    # trim the convolution tail back to the original signal length
    return np.stack([left[:n_samples], right[:n_samples]], axis=1).astype(np.float32)
