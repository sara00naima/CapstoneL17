from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict

from bs_roformer.inference import SafeLoaderWithTuple
from bs_roformer.utils import demix_track, get_model_from_config

from .config import BSROFORMER_CONFIG


def demix_folder(
    input_dir: str,
    output_dir: str,
    model_path: str | None = None,
) -> dict:
    """
    Scans the input directory for .wav files, demixes them all,
    and returns a dictionary grouping the stems by song name.
    """
    print("\n--- Initializing BS-RoFormer Python API ---\n")

    config_path = BSROFORMER_CONFIG
    if model_path is None:
        raise ValueError("model_path must be provided until a default checkpoint path is configured")

    model_path = Path(model_path).resolve()
    in_folder = Path(input_dir).resolve()
    out_folder = Path(output_dir).resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))

    print("\nLoading AI weights into memory...\n")
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"\nModel loaded successfully on: {device}\n")

    all_songs_stems = {}

    for audio_path in in_folder.glob("*.wav"):
        print(f"\nProcessing: {audio_path.name}\n")
        mix, sr = sf.read(str(audio_path))

        original_mono = len(mix.shape) == 1
        if original_mono:
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        with torch.no_grad():
            result, _ = demix_track(config, model, mixture, device)

        stem_name = audio_path.stem
        song_stems = {}

        for instrument, audio in result.items():
            output = audio.T
            if original_mono:
                output = output[:, 0]

            out_file = out_folder / f"{stem_name}_{instrument}.wav"
            sf.write(str(out_file), output, sr, subtype="FLOAT")
            song_stems[instrument] = str(out_file)

        all_songs_stems[stem_name] = song_stems
        print(f"\nFinished demixing {stem_name}!\n")

    print("\n--- Demixing Complete! ---\n")
    return all_songs_stems