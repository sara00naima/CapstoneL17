import numpy as np
from ..core.spherical_harmonics import sh_basis_real

def calculate_decoder_matrix(
    azimuth_rad: np.ndarray, 
    elevation_rad: np.ndarray, 
    radii_m: np.ndarray,
    order: int, 
    normalization: str = "sn3d"
) -> np.ndarray:
    """Calculates the Mode-Matching pseudo-inverse matrix with distance gain compensation."""
    num_speakers = len(azimuth_rad)
    num_channels = (order + 1) ** 2
    
    # The Re-encoding Matrix (Y)
    Y = np.zeros((num_speakers, num_channels), dtype=np.float64)
    for i in range(num_speakers):
        Y[i, :] = sh_basis_real(order, azimuth_rad[i], elevation_rad[i], normalization)
        
    # Return the Decoder Matrix using the Moore-Penrose pseudo-inverse
    decoder_matrix = np.linalg.pinv(Y)
    
    # Distance Gain Compensation (Inverse Distance Law)
    # Attenuate closer speakers so all match the acoustic level of the furthest speaker
    gains = radii_m / np.max(radii_m)
    
    # Apply per-speaker gains (columns of the decoder matrix correspond to speakers)
    decoder_matrix = decoder_matrix * gains[np.newaxis, :]
    
    return decoder_matrix

def calculate_speaker_delays(
    radii_m: np.ndarray, 
    sample_rate: int, 
    speed_of_sound_mps: float = 343.0
) -> np.ndarray:
    """Calculates per-speaker sample delays to time-align the layout to the furthest speaker."""
    max_distance_m = np.max(radii_m)
    delay_seconds = (max_distance_m - radii_m) / speed_of_sound_mps
    delay_samples = np.round(delay_seconds * sample_rate).astype(int)
    return delay_samples

def decode_hoa_to_speakers(
    ambisonic_audio: np.ndarray, 
    decoder_matrix: np.ndarray,
    delay_samples: np.ndarray = None
) -> np.ndarray:
    """Multiplies the 3D sphere by the decoder matrix and applies time alignment."""
    # Matrix Multiplication: (Samples x Channels) @ (Channels x Speakers)
    speaker_feeds = ambisonic_audio @ decoder_matrix
    
    # Distance Delay Compensation
    if delay_samples is not None and np.any(delay_samples > 0):
        num_samples, num_speakers = speaker_feeds.shape
        max_delay = np.max(delay_samples)
        
        # Create an output array padded to fit the longest delay
        aligned_feeds = np.zeros((num_samples + max_delay, num_speakers), dtype=speaker_feeds.dtype)
        
        for i in range(num_speakers):
            d = delay_samples[i]
            aligned_feeds[d : d + num_samples, i] = speaker_feeds[:, i]
            
        return aligned_feeds
        
    return speaker_feeds