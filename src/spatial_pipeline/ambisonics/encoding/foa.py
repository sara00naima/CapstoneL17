import numpy as np

from .hoa import encode_mono_to_hoa

def encode_mono_to_foa(
    signal: np.ndarray,
    azimuth_rad: float,
    elevation_rad: float,
    convention: str = "basic",
) -> np.ndarray:
    if convention == "basic":
        normalization = "sn3d"
    elif convention == "n3d_like":
        normalization = "n3d"
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    return encode_mono_to_hoa(
        signal=signal,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        order=1,
        normalization=normalization,
    )

def sum_hoa_sources(hoa_list: list[np.ndarray]) -> np.ndarray:
    if not hoa_list:
        raise ValueError("Empty source list")
    out = np.zeros_like(hoa_list[0], dtype=np.float32)
    for hoa in hoa_list:
        out += hoa.astype(np.float32)
    return out

def sum_foa_sources(foa_list: list[np.ndarray]) -> np.ndarray:
    return sum_hoa_sources(foa_list)