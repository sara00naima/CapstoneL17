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


def generate_from_recording(
    n_frames: int,
    recording: list[tuple[float, float, float]],
    frame_hop_seconds: float,
) -> list[SphericalPosition]:
    """
    Generates a trajectory from a user-recorded movement (azimuth/elevation
    captured live in the GUI while a "Record Movement" button is held down).

    GUI Integration: While the button is pressed, the GUI samples the mouse
    position on the SceneView and appends (t_seconds, azimuth_deg, elevation_deg)
    tuples to `recording`, where t_seconds is the time elapsed since recording
    started. This works for both continuous drags (e.g. circular motion) and
    discrete clicks (e.g. an instant left/right "teleport"), since we simply
    hold the last known position until a newer sample becomes active —
    no smoothing/interpolation is applied, so sharp jumps stay sharp.

    The recorded gesture is then looped for the entire duration of the
    rendered file: once the per-frame timeline goes past the end of the
    recording, playback wraps back to its start.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the stem being encoded (the recording is
        looped/repeated as needed to cover all of them).
    recording : list[tuple[float, float, float]]
        Captured gesture as (t_seconds, azimuth_deg, elevation_deg) samples,
        sorted by t_seconds, t_seconds starting at (or near) 0.0.
    frame_hop_seconds : float
        Real-world time, in seconds, advanced by each frame (hop_size / sample_rate).
        Used to convert frame index -> timestamp so the loop stays in sync
        regardless of frame_size/hop_size.

    Returns
    -------
    list[SphericalPosition]
        One position per frame, holding/looping the recorded gesture.
    """
    if not recording:
        raise ValueError("recording is empty — nothing was captured")

    # Defensive copy, sorted by time, in case samples arrived slightly out of order.
    samples = sorted(recording, key=lambda s: s[0])

    loop_duration = samples[-1][0]
    if loop_duration <= 0:
        # A single sample (e.g. one click with no drag): treat it as a static
        # anchor point for the whole duration, same behaviour as generate_static.
        azi_deg, ele_deg = samples[0][1], samples[0][2]
        pos = SphericalPosition(azimuth=np.deg2rad(wrap_azimuth_deg(azi_deg)), elevation=np.deg2rad(ele_deg))
        return [pos for _ in range(n_frames)]

    sample_times = [s[0] for s in samples]

    positions = []
    idx = 0  # index of the last sample whose time has already passed, within the current loop
    for frame_i in range(n_frames):
        t = (frame_i * frame_hop_seconds) % loop_duration

        # Advance idx to the last sample with sample_time <= t ("hold" behaviour:
        # no interpolation, so instantaneous jumps in the recording stay instantaneous).
        while idx + 1 < len(sample_times) and sample_times[idx + 1] <= t:
            idx += 1
        # Handle wraparound: if t is smaller than the current sample's time
        # (we looped back to the start), reset idx to the beginning.
        while idx > 0 and sample_times[idx] > t:
            idx -= 1

        _, azi_deg, ele_deg = samples[idx]
        positions.append(
            SphericalPosition(
                azimuth=np.deg2rad(wrap_azimuth_deg(azi_deg)),
                elevation=np.deg2rad(ele_deg),
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