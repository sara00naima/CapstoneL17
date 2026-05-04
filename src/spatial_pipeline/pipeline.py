from .audio_io import load_mono, save_audio
from .ambisonics.core.conventions import deg2rad
from .ambisonics.encoding.foa import encode_mono_to_foa, sum_foa_sources
from .ambisonics.encoding.hoa import encode_mono_to_hoa
import numpy as np
from .ambisonics.decoding.decode_to_speakers import calculate_decoder_matrix, decode_hoa_to_speakers
from .ambisonics.layout.speaker_layout import load_speaker_layout, layout_to_numpy
from .config import MEASUREMENTS_CSV
import soundfile as sf

def encode_stems_to_foa(
    stem_paths: dict[str, str],
    positions_deg: dict[str, tuple[float, float]],
    out_path: str,
    convention: str = "basic",
):
    foa_sources = []
    sr_ref = None
    n_ref = None

    for name, path in stem_paths.items():
        signal, sr = load_mono(path)

        if sr_ref is None:
            sr_ref = sr
            n_ref = len(signal)
        else:
            if sr != sr_ref:
                raise ValueError(f"Sample rate mismatch on stem {name}")
            if len(signal) != n_ref:
                raise ValueError(f"Length mismatch on stem {name}")

        if name not in positions_deg:
            raise KeyError(f"Missing position for stem '{name}'")

        azi_deg, ele_deg = positions_deg[name]
        foa = encode_mono_to_foa(
            signal,
            azimuth_rad=deg2rad(azi_deg),
            elevation_rad=deg2rad(ele_deg),
            convention=convention,
        )
        foa_sources.append(foa)

    bus = sum_foa_sources(foa_sources)
    save_audio(out_path, bus, sr_ref)
    return out_path, sr_ref

def encode_stems_to_hoa(
    stem_paths: dict[str, str],
    positions_deg: dict[str, tuple[float, float]],
    out_path: str,
    order: int = 3,             # 3rd Order - 16 channels
    normalization: str = "sn3d"
):
    hoa_sources = []
    sr_ref = None
    n_ref = None

    for name, path in stem_paths.items():
        signal, sr = load_mono(path)

        if sr_ref is None:
            sr_ref = sr
            n_ref = len(signal)
        else:
            if sr != sr_ref:
                raise ValueError(f"Sample rate mismatch on stem {name}")
            if len(signal) != n_ref:
                raise ValueError(f"Length mismatch on stem {name}")

        if name not in positions_deg:
            raise KeyError(f"Missing position for stem '{name}'")

        azi_deg, ele_deg = positions_deg[name]
        
        hoa = encode_mono_to_hoa(
            signal,
            azimuth_rad=deg2rad(azi_deg),
            elevation_rad=deg2rad(ele_deg),
            order=order,
            normalization=normalization,
        )
        hoa_sources.append(hoa)

    # sum all the layers together
    bus = np.sum(np.stack(hoa_sources), axis=0)
    save_audio(out_path, bus, sr_ref)
    return out_path, sr_ref

def decode_scene_for_ls17(scene_path: str, out_path: str, order: int = 3):
    # load the museum layout
    speakers = load_speaker_layout(MEASUREMENTS_CSV)
    azimuth_rad, elevation_rad, _ = layout_to_numpy(speakers)
    
    # calculate the matrix
    decoder_matrix = calculate_decoder_matrix(azimuth_rad, elevation_rad, order=order)

    # load the audio
    ambisonic_audio, sr = sf.read(scene_path)
    
    # decode and save
    speaker_feeds = decode_hoa_to_speakers(ambisonic_audio, decoder_matrix)
    save_audio(out_path, speaker_feeds.astype(np.float32), sr)
    
    return len(speakers)