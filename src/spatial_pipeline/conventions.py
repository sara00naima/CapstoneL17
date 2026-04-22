import numpy as np

def sph2cart(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    """Convert spherical coordinates (azimuth and elevation in radians) to Cartesian coordinates."""
    x = np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = np.sin(elevation_rad)
    return np.array([x, y, z], dtype=np.float64)

def deg2rad(deg: float) -> float:
    return np.deg2rad(deg)