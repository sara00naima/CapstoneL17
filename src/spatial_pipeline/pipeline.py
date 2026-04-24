from .audio_io import load_mono, save_audio
from .ambisonics.core.conventions import deg2rad
from .ambisonics.encoding.foa import encode_mono_to_foa, sum_foa_sources


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