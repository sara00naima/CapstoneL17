import numpy as np
import soundfile as sf

def load_audio(path: str):
    audio, sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32), sr

def load_mono(path: str):
    audio, sr = load_audio(path)
    if audio.shape[1] == 1:
        return audio[:, 0], sr
    mono = np.mean(audio, axis=1)
    return mono.astype(np.float32), sr

def save_audio(path: str, audio: np.ndarray, sr: int):
    sf.write(path, audio, sr)