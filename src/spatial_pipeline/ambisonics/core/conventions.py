"""
Ambisonics conventions used throughout the project.

Conventions:
- Coordinate system: right-handed
- +X = front
- +Y = left
- +Z = up

Angular convention:
- azimuth = 0 -> front
- azimuth = +90° -> left
- azimuth = -90° -> right

Channel ordering:
- ACN

Normalization:
- SN3D
"""
from enum import Enum
from dataclasses import dataclass
import numpy as np

class ChannelOrdering(Enum):
    ACN = "ACN"
    FUMA = "FuMa"

class Normalization(Enum):
    SN3D = "SN3D"
    N3D = "N3D"
    FUMA = "FUMA"

class CoordinateSystem(Enum):
    RIGHT_HANDED = "right_handed"

@dataclass (frozen=True)
class SphericalPosition:
    azimuth: float
    elevation: float
    distance: float = 1.0

    def __post_init__(self):
        validate_angles(self.azimuth, self.elevation)

        if self.distance <= 0:
            raise ValueError(
                "Distance must be positive"
            )

#GLOBAL CONVENTIONS
DEFAULT_ORDERING = ChannelOrdering.ACN
DEFAULT_NORMALIZATION = Normalization.SN3D
DEFAULT_COORD_SYSTEM = CoordinateSystem.RIGHT_HANDED

#ANGULAR CONVENTIONS
FRONT_AZIMUTH = 0.0
LEFT_AZIMUTH = np.pi / 2
RIGHT_AZIMUTH = -np.pi / 2
BACK_AZIMUTH = np.pi

#ELEVATION 
HORIZON_ELEVATION = 0.0
UP_ELEVATION = np.pi / 2
DOWN_ELEVATION = -np.pi / 2

#ACN mappng
FOA_CHANNELS_ACN = {
    0: "W",
    1: "Y",
    2: "Z",
    3: "X",
}

#reverse mapping for convenience: we can write 
#audio[FOA_CHANNEL_INDEX["X"]]
FOA_CHANNEL_INDEX = {
    "W": 0,
    "Y": 1,
    "Z": 2,
    "X": 3,
}

# returns channel's name (letter)
def channel_name(index: int) -> str:
    return FOA_CHANNELS_ACN[index]

# returns channel's index 
def channel_index(name: str) -> int:
    return FOA_CHANNEL_INDEX[name]

# azimuth normalization
def normalize_azimuth(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

# validate values of azimuth and elevation angles
def validate_angles(azimuth, elevation):

    if not -np.pi <= azimuth <= np.pi:
        raise ValueError(
            "Azimuth must be between -π and +π radians"
        )

    if not -np.pi / 2 <= elevation <= np.pi / 2:
        raise ValueError(
            "Elevation must be between -π/2 and +π/2 radians"
        )

# transition from spherical coordinates to cartesian coordinates
def sph2cart(position: SphericalPosition) -> np.ndarray:

    az = position.azimuth
    el = position.elevation
    r = position.distance

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    return np.array([x, y, z], dtype=np.float64)

# converts degrees to radians
def deg2rad(deg: float) -> float:
    return np.deg2rad(deg)

# converts radians to degrees
def rad2deg(radians) -> float:
    return np.rad2deg(radians)