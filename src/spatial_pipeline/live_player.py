"""Real-time binaural preview using direct HRTF convolution per stem."""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
import sounddevice as sd
import sofar
import soundfile as sf

from .ambisonics.core.conventions import deg2rad
from .binaural import nearest_hrtf_index


class LivePlayer:
    """
    Streams all active stems in real-time with binaural spatialisation.
    Reads azimuth / elevation / gain / mute directly from AppState on every
    audio callback, so any change in the GUI is heard within one block
    (~23 ms at 44100 Hz) without re-generating the output file.

    Rendering uses the same FFT-based convolution as binaural.py, but in a
    stateful, block-by-block fashion instead of a single offline pass:

      For each block and each stem:
        1. Grab `BLOCKSIZE` samples from the looping stem audio.
        2. Look up the nearest HRTF measurement to the current position
           (same great-circle nearest-neighbour as nearest_hrtf_index).
        3. Convolve via FFT (same overlap-add principle as frame_processing.py,
           but stateful: the convolution tail is carried over to the next block).
        4. Accumulate L/R into the output buffer.
    """

    BLOCKSIZE = 1024

    def __init__(self, state, sofa_path: str):
        self._state = state

        # Load HRTF — same loading path as binaural.py
        hrtf = sofar.read_sofa(sofa_path)
        self._sofa_az = deg2rad(hrtf.SourcePosition[:, 0])  # CCW positive (left = +)
        self._sofa_el = deg2rad(hrtf.SourcePosition[:, 1])
        self._hrirs = hrtf.Data_IR.astype(np.float64)       # (M, 2, N)
        self._ir_len = self._hrirs.shape[2]

        # FFT size for OLA convolution — same formula as binaural.py
        self._n_fft = next_fast_len(self.BLOCKSIZE + self._ir_len - 1)

        # Pre-compute HRIR FFTs for every SOFA position to avoid recomputing
        # them inside the real-time callback (same rfft as binaural.py uses)
        self._hrir_ffts = np.array([
            [rfft(self._hrirs[m, 0], n=self._n_fft),
             rfft(self._hrirs[m, 1], n=self._n_fft)]
            for m in range(self._hrirs.shape[0])
        ])  # (M, 2, n_fft//2 + 1) complex128

        # Load stems into memory as mono float64
        self._stems = {}     # name -> (n_samples,) float64
        self._read_pos = {}  # name -> int   (sample cursor, loops at end)
        # Per-stem OLA tail: the (ir_len - 1) samples that spill past the
        # current block boundary and must be added to the start of the next block.
        # This is the stateful equivalent of the accumulation buffer in overlap_add().
        self._tails = {}     # name -> (ir_len - 1, 2) float64
        self._sr = None

        for src in state.sources:
            if not src.wav_path:
                continue
            audio, sr = sf.read(src.wav_path, dtype='float32')
            if self._sr is None:
                self._sr = sr
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._stems[src.name] = audio.astype(np.float64)
            self._read_pos[src.name] = 0
            self._tails[src.name] = np.zeros((self._ir_len - 1, 2), dtype=np.float64)

        if not self._stems:
            raise ValueError(
                "No stems loaded — assign a WAV file to at least one source before previewing."
            )

        self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        out = np.zeros((frames, 2), dtype=np.float64)

        for src in self._state.sources:
            if src.mute or src.name not in self._stems:
                continue

            audio = self._stems[src.name]
            pos = self._read_pos[src.name]

            # Read `frames` samples with seamless looping
            chunk = np.empty(frames, dtype=np.float64)
            remaining, offset = frames, 0
            while remaining > 0:
                take = min(len(audio) - pos, remaining)
                chunk[offset:offset + take] = audio[pos:pos + take]
                offset += take
                remaining -= take
                pos = (pos + take) % len(audio)
            self._read_pos[src.name] = pos

            if src.gain_db != 0.0:
                chunk *= 10.0 ** (src.gain_db / 20.0)

            # GUI azimuth uses screen convention (right = +).
            # SOFA and Ambisonics use CCW (left = +), so negate — same as _do_generate.
            az_rad = deg2rad(-src.azimuth)
            el_rad = deg2rad(src.elevation)
            idx = nearest_hrtf_index(az_rad, el_rad, self._sofa_az, self._sofa_el)
            hrir_fft = self._hrir_ffts[idx]  # (2, n_fft//2 + 1)

            # FFT convolution — identical to the rfft/irfft path in binaural.py
            chunk_fft = rfft(chunk, n=self._n_fft)
            conv_len = frames + self._ir_len - 1
            left  = irfft(chunk_fft * hrir_fft[0], n=self._n_fft)[:conv_len]
            right = irfft(chunk_fft * hrir_fft[1], n=self._n_fft)[:conv_len]
            conv = np.stack([left, right], axis=1)  # (conv_len, 2)

            # Overlap-add: add the saved tail from the previous block, then
            # save the new tail for the next block (the stateful part).
            tail = self._tails[src.name]
            conv[:len(tail)] += tail
            self._tails[src.name] = conv[frames:].copy()

            out += conv[:frames]

        # Soft-limit to avoid hard clipping
        peak = np.max(np.abs(out))
        if peak > 0.9:
            out *= 0.9 / peak

        outdata[:] = out.astype(np.float32)

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=2,
            dtype='float32',
            blocksize=self.BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
