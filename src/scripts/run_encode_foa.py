from spatial_pipeline.pipeline import encode_stems_to_foa

stem_paths = {
    "vocals": "data/stems/vocals.wav",
    "drums": "data/stems/drums.wav",
    "bass": "data/stems/bass.wav",
    "other": "data/stems/other.wav",
}

positions_deg = {
    "vocals": (0.0, 0.0),
    "drums": (-30.0, 0.0),
    "bass": (25.0, 0.0),
    "other": (110.0, 10.0),
}

encode_stems_to_foa(
    stem_paths=stem_paths,
    positions_deg=positions_deg,
    out_path="data/rendered/scene_foa.wav",
    convention="basic"
)