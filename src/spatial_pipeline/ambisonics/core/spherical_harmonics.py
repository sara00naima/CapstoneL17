import math
import numpy as np
from scipy.special import lpmv


def acn_index(n: int, m: int) -> int:
    return n * n + n + m


def num_harmonics(order: int) -> int:
    return (order + 1) ** 2


def sh_basis_real(
    order: int,
    azimuth_rad: float,
    elevation_rad: float,
    normalization: str = "n3d",
) -> np.ndarray:
    phi = azimuth_rad
    theta = np.pi / 2.0 - elevation_rad

    y = np.zeros(num_harmonics(order), dtype=np.float64)
    cos_theta = np.cos(theta)

    for n in range(order + 1):
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            abs_m = abs(m)

            p_nm = lpmv(abs_m, n, cos_theta)
            p_nm *= (-1) ** abs_m  # remove Condon-Shortley phase from scipy

            if normalization == "n3d":
                norm = math.sqrt(
                    (2 * n + 1)
                    * math.factorial(n - abs_m)
                    / math.factorial(n + abs_m)
                )
            elif normalization == "sn3d":
                norm = math.sqrt(
                    math.factorial(n - abs_m)
                    / math.factorial(n + abs_m)
                )
            else:
                raise ValueError(f"Unsupported normalization: {normalization}")

            if m > 0:
                y[idx] = math.sqrt(2.0) * norm * p_nm * np.cos(m * phi)
            elif m < 0:
                y[idx] = math.sqrt(2.0) * norm * p_nm * np.sin(abs_m * phi)
            else:
                y[idx] = norm * p_nm

    return y
