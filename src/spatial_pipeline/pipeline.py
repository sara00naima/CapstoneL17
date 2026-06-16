from .audio_io import load_mono, save_audio
from .ambisonics.core.conventions import SphericalPosition, deg2rad
from .ambisonics.encoding.foa import encode_mono_to_foa, sum_foa_sources
from .binaural import render_binaural, render_ls17_binaural
from .ambisonics.encoding.hoa import encode_mono_to_hoa
import numpy as np
from .ambisonics.decoding.decode_to_speakers import calculate_decoder_matrix, decode_hoa_to_speakers
from .ambisonics.layout.speaker_layout import load_speaker_layout, layout_to_numpy
from .frame_processing import split_frames
from .frame_ambisonics import process_hoa_frames
from .ambisonics.core.trajectories import generate_static
from .config import MEASUREMENTS_CSV
import soundfile as sf


def encode_stems_to_foa(
    stem_paths: dict[str, str],         # { instrument: path_to_wav }
    positions_deg: dict[str, tuple[float, float]], # { instrument: (azimuth, elevation) }
    out_path: str,                       # where to write the FOA bus
    convention: str = "basic",           # SH normalization convention
):
    foa_sources = []  # accumulates the encoded FOA signal for each stem
    sr_ref = None     # sample rate of the first stem, used to validate all others
    n_ref = None      # sample count of the first stem, used to validate all others

    for name, path in stem_paths.items():
        signal, sr = load_mono(path)

        if sr_ref is None:
            # First stem sets the reference sample rate and length
            sr_ref = sr
            n_ref = len(signal)
        else:
            # All subsequent stems must match
            if sr != sr_ref:
                raise ValueError(f"Sample rate mismatch on stem {name}")
            if len(signal) != n_ref:
                raise ValueError(f"Length mismatch on stem {name}")

        # Each stem must have a declared position in space
        if name not in positions_deg:
            raise KeyError(f"Missing position for stem '{name}'")

        # Convert degrees to radians and wrap into a SphericalPosition object
        azi_deg, ele_deg = positions_deg[name]
        position = SphericalPosition(
            azimuth=deg2rad(azi_deg),
            elevation=deg2rad(ele_deg),
        )

        # Encode the mono signal into 4-channel FOA using the source direction
        foa = encode_mono_to_foa(
            signal,
            position=position,
            convention=convention,
        )
        foa_sources.append(foa)

    # Sum all encoded FOA sources into a single 4-channel ambisonic bus
    bus = sum_foa_sources(foa_sources)

    save_audio(out_path, bus, sr_ref)
    return out_path, sr_ref


def encode_stems_to_hoa(
    stem_paths: dict[str, str],                    # { instrument: path_to_wav }
    positions_deg: dict[str, tuple[float, float]], # { instrument: (azimuth, elevation) }
    out_path: str,                                 # where to write the HOA bus
    order: int = 3,                                # 3rd Order - 16 channels
    normalization: str = "sn3d",                   # AmbiX standard normalization
    trajectory_fn=generate_static,                 # defaults to no movement
    trajectories: dict[str, list] | None = None,
):
    hoa_sources = []  # accumulates the encoded HOA signal for each stem
    sr_ref = None     # reference sample rate, set from the first stem
    n_ref = None      # reference sample count, set from the first stem

    # Frame processing parameters — defined once here and passed through to
    # process_hoa_frames, so n_frames is always computed with the same values.
    frame_size = 1024
    hop_size = 512

    for name, path in stem_paths.items():
        signal, sr = load_mono(path)

        if sr_ref is None:
            # First stem sets the reference sample rate and length
            sr_ref = sr
            n_ref = len(signal)
        else:
            # All subsequent stems must match
            if sr != sr_ref:
                raise ValueError(f"Sample rate mismatch on stem {name}")
            if len(signal) != n_ref:
                raise ValueError(f"Length mismatch on stem {name}")

        # Each stem must have a declared position in space
        if name not in positions_deg:
            raise KeyError(f"Missing position for stem '{name}'")

        # Compute n_frames using the same frame_size and hop_size that
        # process_hoa_frames will use — guarantees they always match.
        n_frames = len(split_frames(signal, frame_size, hop_size))

        if trajectories is not None:
            if name not in trajectories:
                raise KeyError(f"Missing trajectory for stem '{name}'")
            trajectory = trajectories[name]
        else:
            azi_deg, ele_deg = positions_deg[name]
            trajectory = trajectory_fn(n_frames, azi_deg, ele_deg)

        # Encode the mono signal into HOA frame by frame,
        # applying the trajectory position at each frame.
        hoa = process_hoa_frames(
            signal=signal,
            positions=trajectory,
            order=order,
            frame_size=frame_size,
            hop_size=hop_size,
            normalization=normalization,
        )
        hoa_sources.append(hoa)

    if not hoa_sources:
        raise ValueError("Empty source list")

    # Sum all encoded HOA sources into a single ambisonic bus
    bus = np.zeros_like(hoa_sources[0], dtype=np.float32)
    for hoa in hoa_sources:
        bus += hoa.astype(np.float32)

    save_audio(out_path, bus, sr_ref)
    return out_path, sr_ref


def render_binaural_scene(
    scene_path: str,  # path to the encoded HOA scene WAV
    sofa_path: str,   # path to the SOFA HRTF file
    out_path: str,    # where to write the binaural stereo WAV
    order: int = 3,
) -> str:
    """
    Renders a binaural stereo WAV from an ambisonic scene file.
    Uses the virtual loudspeaker method with HRTFs from a SOFA file.
    Returns the output path.
    """
    # Load the encoded ambisonic scene
    ambisonic_audio, sr = sf.read(scene_path)

    # Render to binaural stereo: (samples, 2)
    binaural = render_binaural(ambisonic_audio, sofa_path, order=order)

    # Save the stereo output
    save_audio(out_path, binaural, sr)
    return out_path


def render_ls17_binaural_scene(
    scene_path: str,
    sofa_path: str,
    out_path: str,
    order: int = 3,
) -> str:
    """
    Renders binaural stereo by routing through the LS17 museum decoder first,
    then applying per-speaker HRTFs. Simulates the museum listening experience
    on headphones without needing physical speakers.
    """
    ambisonic_audio, sr = sf.read(scene_path)
    binaural = render_ls17_binaural(
        ambisonic_audio, sofa_path, str(MEASUREMENTS_CSV), order=order
    )
    save_audio(out_path, binaural, sr)
    return out_path


def decode_scene_for_ls17(scene_path: str, out_path: str, order: int = 3):
    """
    Decodes an ambisonic scene to the speaker layout defined in
    MEASUREMENTS_CSV (Violin museum), writing one audio channel per loudspeaker.
    Returns the number of speakers in the layout.
    """
    # Load the museum speaker layout from the measurements CSV
    speakers = load_speaker_layout(MEASUREMENTS_CSV)

    # Unpack speaker positions into numpy arrays for the decoder
    azimuth_rad, elevation_rad, _ = layout_to_numpy(speakers)

    # Calculate the mode-matching decoder matrix for this layout
    decoder_matrix = calculate_decoder_matrix(azimuth_rad, elevation_rad, order=order)

    # Load the encoded ambisonic scene
    ambisonic_audio, sr = sf.read(scene_path)

    # Decode to one audio channel per loudspeaker and save
    speaker_feeds = decode_hoa_to_speakers(ambisonic_audio, decoder_matrix)
    save_audio(out_path, speaker_feeds.astype(np.float32), sr)

    return len(speakers)

def decode_scene_for_layout(scene_path: str, out_path: str, layout_csv: str, order: int = 3):
    """
    Decodes an ambisonic scene to an arbitrary speaker layout defined by a CSV file.
    The CSV is loaded with load_speaker_layout (same format as MEASUREMENTS_CSV).
    Writes one audio channel per loudspeaker.
    Returns the number of speakers in the layout.
    """
    speakers = load_speaker_layout(layout_csv)
    azimuth_rad, elevation_rad, _ = layout_to_numpy(speakers)
    decoder_matrix = calculate_decoder_matrix(azimuth_rad, elevation_rad, order=order)
    ambisonic_audio, sr = sf.read(scene_path)
    speaker_feeds = decode_hoa_to_speakers(ambisonic_audio, decoder_matrix)
    save_audio(out_path, speaker_feeds.astype(np.float32), sr)
    return len(speakers)
