import numpy as np
from ..core.spherical_harmonics import sh_basis_real

def calculate_decoder_matrix(
    azimuth_rad: np.ndarray, 
    elevation_rad: np.ndarray, 
    order: int, 
    normalization: str = "sn3d"
) -> np.ndarray:
    """Calculates the Mode-Matching pseudo-inverse matrix for a given speaker layout."""
    num_speakers = len(azimuth_rad)
    num_channels = (order + 1) ** 2
    
    # the Re-encoding Matrix (Y)
    Y = np.zeros((num_speakers, num_channels), dtype=np.float64)
    for i in range(num_speakers):
        Y[i, :] = sh_basis_real(order, azimuth_rad[i], elevation_rad[i], normalization)
    # return the Decoder Matrix using the Moore-Penrose pseudo-inverse
    return np.linalg.pinv(Y)

def decode_hoa_to_speakers(
    ambisonic_audio: np.ndarray, 
    decoder_matrix: np.ndarray
) -> np.ndarray:
    """Multiplies the 3D sphere by the decoder matrix to get individual speaker feeds."""
    # Matrix Multiplication: (Samples x Channels) @ (Channels x Speakers)
    return ambisonic_audio @ decoder_matrix