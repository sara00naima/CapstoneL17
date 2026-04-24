import numpy as np

from ..core.conventions import sph2cart


def encode_mono_to_foa(
    signal: np.ndarray,
    azimuth_rad: float,
    elevation_rad: float,
    convention: str = "basic",
) -> np.ndarray:
    """
    Input:
        signal: shape (samples,)
        convention: either "basic" or "n3d_like"
    Output:
        foa: shape (samples, 4) ordered as [W, Y, Z, X]
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    x, y, z = sph2cart(azimuth_rad, elevation_rad)

    if convention == "basic":
        w_gain = 1.0
        x_gain = x
        y_gain = y
        z_gain = z
    elif convention == "n3d_like":
        w_gain = 1.0
        g = np.sqrt(3.0)
        x_gain = g * x
        y_gain = g * y
        z_gain = g * z
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    foa = np.stack(
        [
            signal * w_gain,
            signal * y_gain,
            signal * z_gain,
            signal * x_gain,
        ],
        axis=-1,
    )

    return foa


def sum_foa_sources(foa_list: list[np.ndarray]) -> np.ndarray:
    if not foa_list:
        raise ValueError("Empty source list")

    out = np.zeros_like(foa_list[0], dtype=np.float32)
    for foa in foa_list:
        out += foa.astype(np.float32)
    return out