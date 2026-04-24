import numpy as np

from ..core.spherical_harmonics import sh_basis_real

def encode_mono_to_hoa(
    signal: np.ndarray,
    azimuth_rad: float,
    elevation_rad: float,
    order: int,
    normalization: str = "n3d",
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    gains = sh_basis_real(
        order=order,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        normalization=normalization,
    ).astype(np.float32)

    return signal[:, None] * gains[None, :]