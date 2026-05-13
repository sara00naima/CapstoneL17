# High order ambisonics
import numpy as np

from ..core.spherical_harmonics import sh_basis_real
from ..core.conventions import SphericalPosition

def encode_mono_to_hoa(
    signal: np.ndarray,
    position: SphericalPosition,
    order: int,
    normalization: str = "n3d",
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    gains = sh_basis_real(
        order=order,
        azimuth_rad=position.azimuth,
        elevation_rad=position.elevation,
        normalization=normalization,
    ).astype(np.float32)

    return signal[:, None] * gains[None, :]