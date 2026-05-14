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

    # Path to the model architecture config (yaml), defined in .config module
    config_path = BSROFORMER_CONFIG

    # No default checkpoint is bundled, caller must supply one explicitly
    if model_path is None:
        raise ValueError("model_path must be provided until a default checkpoint path is configured")

    # Resolve all paths to absolute to avoid ambiguity regardless of cwd
    model_path = Path(model_path).resolve()
    in_folder  = Path(input_dir).resolve()
    out_folder = Path(output_dir).resolve()

    # Create output directory if it doesn't exist (including any missing parents)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Load the BS-RoFormer architecture config from yaml.
    # SafeLoaderWithTuple extends PyYAML's SafeLoader to handle Python tuples
    # which appear in the config (e.g. for kernel sizes or shape definitions)
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))

    print("\nLoading AI weights into memory...\n")

    # Instantiate the BS-RoFormer model architecture from the config,
    # then load the pre-trained weights into it
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set model to inference mode
    model.eval()

    print(f"\nModel loaded successfully on: {device}\n")

    # Accumulates results across all songs:
    # { "song_name": { "vocals": "/path/to/file.wav", "drums": "...", ... } }
    all_songs_stems = {}

    for audio_path in in_folder.glob("*.wav"):
        print(f"\nProcessing: {audio_path.name}\n")

        # Read audio as numpy array (samples × channels) and sample rate
        mix, sr = sf.read(str(audio_path))

        # BS-RoFormer expects stereo input, if the file is mono,
        # duplicate it into a fake stereo signal so the model doesn't break
        original_mono = len(mix.shape) == 1
        if original_mono:
            mix = np.stack([mix, mix], axis=-1)

        # Transpose from (samples × channels) to (channels × samples)
        # which is the convention expected by the model
        mixture = torch.tensor(mix.T, dtype=torch.float32)

        # Run the source separation, no_grad disables gradient tracking
        # since we're only doing inference, not training
        with torch.no_grad():
            result, _ = demix_track(config, model, mixture, device)

        # Use the filename (without extension) as the song identifier
        stem_name = audio_path.stem
        song_stems = {}

        for instrument, audio in result.items():
            # Transpose back from (channels × samples) to (samples × channels)
            output = audio.T

            # If the original file was mono, discard the duplicated channel
            # and return a 1D array to match the input format
            if original_mono:
                output = output[:, 0]

            # Write each separated stem as a 32-bit float wav file
            out_file = out_folder / f"{stem_name}_{instrument}.wav"
            sf.write(str(out_file), output, sr, subtype="FLOAT")

            # Store the output path keyed by instrument name
            song_stems[instrument] = str(out_file)

        all_songs_stems[stem_name] = song_stems
        print(f"\nFinished demixing {stem_name}!\n")

    print("\n--- Demixing Complete! ---\n")

    return all_songs_stems