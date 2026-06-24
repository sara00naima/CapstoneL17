import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
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
    radii_m = hrtf.SourcePosition[:, 2]
    decoder_matrix = calculate_decoder_matrix(
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        radii_m=radii_m,
        order=order,
        normalization=normalization,
    )

    # Decode the ambisonic bus to virtual speaker feeds: (samples, M)
    virtual_feeds = decode_hoa_to_speakers(ambisonic_audio, decoder_matrix)

    n_samples = ambisonic_audio.shape[0]
    ir_len    = hrirs.shape[2]
    N_fft     = next_fast_len(n_samples + ir_len - 1)

    # Accumulate left/right in the frequency domain: one IFFT per ear instead
    # of one per virtual speaker (M × 2 → 2 IFFTs total).
    left_fft  = np.zeros(N_fft // 2 + 1, dtype=np.complex128)
    right_fft = np.zeros(N_fft // 2 + 1, dtype=np.complex128)

    for i in range(virtual_feeds.shape[1]):
        feed_fft   = rfft(virtual_feeds[:, i], n=N_fft)
        left_fft  += feed_fft * rfft(hrirs[i, 0, :], n=N_fft)
        right_fft += feed_fft * rfft(hrirs[i, 1, :], n=N_fft)

    left  = irfft(left_fft,  n=N_fft)[:n_samples]
    right = irfft(right_fft, n=N_fft)[:n_samples]
    return np.stack([left, right], axis=1).astype(np.float32)


def nearest_hrtf_index(
    az: float,
    el: float,
    sofa_az: np.ndarray,
    sofa_el: np.ndarray,
) -> int:
    # Great-circle similarity: argmax of unit-vector dot product
    dot = (np.sin(el) * np.sin(sofa_el) +
           np.cos(el) * np.cos(sofa_el) * np.cos(az - sofa_az))
    return int(np.argmax(dot))


def render_ls17_binaural(
    ambisonic_audio: np.ndarray,
    sofa_path: str,
    csv_path: str,
    order: int = 3,
    normalization: str = "sn3d",
) -> np.ndarray:  # (samples, 2) float32 [left, right]
    """
    Renders binaural stereo by first decoding the HOA bus through the LS17
    museum speaker layout, then convolving each speaker feed with the nearest
    HRTF from the SOFA file.  This simulates sitting at the sweet spot of the
    museum system on headphones, making the LS17 decoder auditionable without
    physical speakers.
    """
    from .ambisonics.layout.speaker_layout import load_speaker_layout, layout_to_numpy

    speakers = load_speaker_layout(csv_path)
    az_rad, el_rad, cartesian = layout_to_numpy(speakers)
    radii_m = np.linalg.norm(cartesian, axis=1)

    # Use explicit keyword arguments to avoid passing the string "sn3d" into the 'order' integer parameter
    decoder_matrix = calculate_decoder_matrix(
        azimuth_rad=az_rad, 
        elevation_rad=el_rad, 
        radii_m=radii_m, 
        order=order, 
        normalization=normalization
    )
    speaker_feeds = decode_hoa_to_speakers(ambisonic_audio, decoder_matrix)  # (samples, 17)

    hrtf = sofar.read_sofa(sofa_path)
    sofa_az = deg2rad(hrtf.SourcePosition[:, 0])
    sofa_el = deg2rad(hrtf.SourcePosition[:, 1])
    hrirs = hrtf.Data_IR  # (M, 2, N)

    n_samples = ambisonic_audio.shape[0]
    ir_len    = hrirs.shape[2]
    N_fft     = next_fast_len(n_samples + ir_len - 1)

    left_fft  = np.zeros(N_fft // 2 + 1, dtype=np.complex128)
    right_fft = np.zeros(N_fft // 2 + 1, dtype=np.complex128)

    for i in range(len(speakers)):
        idx        = nearest_hrtf_index(az_rad[i], el_rad[i], sofa_az, sofa_el)
        feed_fft   = rfft(speaker_feeds[:, i], n=N_fft)
        left_fft  += feed_fft * rfft(hrirs[idx, 0, :], n=N_fft)
        right_fft += feed_fft * rfft(hrirs[idx, 1, :], n=N_fft)

    left  = irfft(left_fft,  n=N_fft)[:n_samples]
    right = irfft(right_fft, n=N_fft)[:n_samples]
    return np.stack([left, right], axis=1).astype(np.float32)
