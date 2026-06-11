import math
import numpy as np
from scipy.special import lpmv


def acn_index(n: int, m: int) -> int:
    """Return ACN channel index for order n and degree m."""
    return n * n + n + m


def num_harmonics(order: int) -> int:
    """Total number of Ambisonic channels up to given order."""
    return (order + 1) ** 2


def sh_basis_real(
    order: int,
    azimuth_rad: float,
    elevation_rad: float,
    normalization: str = "sn3d",
) -> np.ndarray:
    """
    Compute real spherical harmonics in ACN channel order.

    Parameters
    ----------
    order : int
        Ambisonic order.
    azimuth_rad : float
        Azimuth in radians.
    elevation_rad : float
        Elevation in radians.
    normalization : str
        "sn3d" or "n3d".

    Returns
    -------
    np.ndarray
        Real SH basis vector in ACN ordering.
    """
    normalization = normalization.lower()
    if normalization not in {"sn3d", "n3d"}:
        raise ValueError(f"Unsupported normalization: {normalization}")

    phi = azimuth_rad
    theta = np.pi / 2.0 - elevation_rad
    cos_theta = np.cos(theta)

    y = np.zeros(num_harmonics(order), dtype=np.float64)

    for n in range(order + 1):
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            abs_m = abs(m)

            p_nm = lpmv(abs_m, n, cos_theta)
            p_nm *= (-1) ** abs_m

            factorial_ratio = (
                math.factorial(n - abs_m) / math.factorial(n + abs_m)
            )

            if normalization == "sn3d":
                if m == 0:
                    norm = math.sqrt(factorial_ratio)
                else:
                    norm = math.sqrt(2.0 * factorial_ratio)
            else:  # n3d
                if m == 0:
                    norm = math.sqrt((2 * n + 1) * factorial_ratio)
                else:
                    norm = math.sqrt(2.0 * (2 * n + 1) * factorial_ratio)

            if m > 0:
                y[idx] = norm * p_nm * math.cos(m * phi)
            elif m < 0:
                y[idx] = norm * p_nm * math.sin(abs_m * phi)
            else:
                y[idx] = norm * p_nm

    return y


def normalization_gain(n: int, from_norm: str, to_norm: str) -> float:
    """
    Gain to convert one Ambisonic order from one normalization to another.

    For a fixed order n:
    N3D = SN3D * sqrt(2n + 1)
    SN3D = N3D / sqrt(2n + 1)
    """
    from_norm = from_norm.lower()
    to_norm = to_norm.lower()

    if from_norm == to_norm:
        return 1.0
    if from_norm == "sn3d" and to_norm == "n3d":
        return math.sqrt(2 * n + 1)
    if from_norm == "n3d" and to_norm == "sn3d":
        return 1.0 / math.sqrt(2 * n + 1)

    raise ValueError(f"Unsupported normalization conversion: {from_norm} -> {to_norm}")


def convert_normalization(
    coeffs: np.ndarray,
    order: int,
    from_norm: str,
    to_norm: str,
) -> np.ndarray:
    """
    Convert Ambisonic coefficients between SN3D and N3D, preserving ACN order.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64).copy()

    expected = num_harmonics(order)
    if coeffs.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} coefficients for order {order}, got {coeffs.shape[0]}"
        )

    for n in range(order + 1):
        g = normalization_gain(n, from_norm, to_norm)
        start = n * n
        end = (n + 1) * (n + 1)
        coeffs[start:end] *= g

    return coeffs


if __name__ == "__main__":
    order = 1
    az = 0.0
    el = 0.0

    y_sn3d = sh_basis_real(order, az, el, normalization="sn3d")
    y_n3d = sh_basis_real(order, az, el, normalization="n3d")

    print("SN3D:", y_sn3d)
    print("N3D: ", y_n3d)

    y_sn3d_to_n3d = convert_normalization(y_sn3d, order, "sn3d", "n3d")
    print("SN3D -> N3D:", y_sn3d_to_n3d)

    print("Match N3D:", np.allclose(y_n3d, y_sn3d_to_n3d))