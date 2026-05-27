import numpy as np
from .conventions import SphericalPosition, wrap_azimuth_deg

def generate_static(n_frames: int, azimuth_deg: float, elevation_deg: float) -> list[SphericalPosition]:
    """
    Generates a list of identical positions (no movement)
    GUI Integration: Used when a user drags an instrument to a specific spot 
    and leaves it there. The anchor point remains constant for the whole song.
    """
    pos = SphericalPosition(azimuth=np.deg2rad(azimuth_deg), elevation=np.deg2rad(elevation_deg))
    return [pos for _ in range(n_frames)]


def generate_orbit(n_frames: int, start_azi_deg: float, rotations: float = 1.0, elevation_deg: float = 0.0) -> list[SphericalPosition]:
    """
    Generates a circular 360-degree orbit around the listener.
    GUI Integration: Triggered by an 'Orbit' effect dropdown. 
    start_azi_deg is the anchor point where the user placed the instrument.
    """
    # Calculate the total degrees to travel based on desired rotations
    total_degrees = 360.0 * rotations
    
    # Create an array of smoothly increasing angles over the total frames
    azimuths = np.linspace(start_azi_deg, start_azi_deg + total_degrees, n_frames)
    
    positions = []
    for azi in azimuths:
        # Wrap the angle back to the strict [-180, 180] boundary
        wrapped_azi = wrap_azimuth_deg(azi)
        positions.append(
            SphericalPosition(
                azimuth=np.deg2rad(wrapped_azi),
                elevation=np.deg2rad(elevation_deg)
            )
        )
    return positions


def generate_arc_flyover(n_frames: int, azimuth_deg: float, start_ele_deg: float = -90.0, end_ele_deg: float = 90.0) -> list[SphericalPosition]:
    """
    Generates an elevation sweep (e.g., flying over the listener's head).
    GUI Integration: Triggered by a 'Flyover' effect dropdown.
    azimuth_deg acts as the anchor track (e.g., 0 is front-to-back, 90 is side-to-side).
    """
    # Create an array of smoothly changing elevation angles
    elevations = np.linspace(start_ele_deg, end_ele_deg, n_frames)
    
    positions = []
    for ele in elevations:
        positions.append(
            SphericalPosition(
                azimuth=np.deg2rad(wrap_azimuth_deg(azimuth_deg)),
                elevation=np.deg2rad(ele) # Convention validation ensures it stays within [-90, 90]
            )
        )
    return positions


def generate_bounce(n_frames: int, center_azi_deg: float, width_deg: float = 45.0, bounces: float = 10.0, elevation_deg: float = 0.0) -> list[SphericalPosition]:
    """
    Generates a rhythmic left-to-right panning motion around a center point.
    GUI Integration: Triggered by a 'Bounce' effect dropdown.
    center_azi_deg is where the user dropped the instrument; width determines how far it swings.
    """
    # Create a time array from 0 to 2*PI*bounces to feed into a sine wave
    t = np.linspace(0, 2 * np.pi * bounces, n_frames)
    
    # Calculate oscillating azimuths using the sine wave (scales from -1 to 1, multiplied by width)
    azimuths = center_azi_deg + (np.sin(t) * width_deg)
    
    positions = []
    for azi in azimuths:
        wrapped_azi = wrap_azimuth_deg(azi)
        positions.append(
            SphericalPosition(
                azimuth=np.deg2rad(wrapped_azi),
                elevation=np.deg2rad(elevation_deg)
            )
        )
    return positions