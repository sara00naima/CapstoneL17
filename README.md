# Sonara — Ambisonic Render Engine

**Sonara** is a Python-based spatial audio pipeline that transforms a stereo music track into an immersive three-dimensional listening experience. It automates the full processing chain, starting from source separation to Ambisonic encoding and binaural rendering through an interactive graphical interface.

The pipeline separates a mixed audio track into individual stems (vocals, drums, bass, guitar, piano, other) using a BS-RoFormer deep learning model, places each stem at a user-defined position in a spherical coordinate space, and encodes the result into High-Order Ambisonics (HOA) up to 7th order following the AmbiX convention (ACN channel ordering, SN3D normalisation). The final scene can be rendered either binaurally via a user-supplied HRTF or decoded to a custom loudspeaker layout.

## Features

- **Interactive scene view** — drag sources around a top-down azimuth map, adjust elevation and gain per stem;
- **Movement recording** — capture a freehand gesture on the scene view and loop it for the full render;
- **Live preview** — audition the spatial mix in real time before committing to a render;
- **Configurable output** — binaural headphone render or multi-channel speaker layout decode;
- **Dark-themed GUI** built with Tkinter and PIL-rendered custom controls.


## Project structure

```text
CapstoneL17/
├── src/
│   ├── spatial_pipeline/     # main library (encoding, decoding, binaural, demix...)
│   ├── gui/                  # graphical interface (Tkinter)
│   ├── scripts/              # CLI scripts, runnable stage by stage
│   └── evaluation/           # evaluation / analysis tools (spatial, reproduction, DOA)
├── Demixing BS-RoF/          # subproject / checkpoint for the BS-RoFormer model
├── hrtf/                     # SOFA files containing the HRTFs for binaural rendering
├── assets/                   # static GUI assets
├── tests/                    # automated tests
├── outputs/                  # generated outputs (created automatically, not versioned)
├── measurements_transcription.csv   # museum loudspeaker layout (17 channels)
├── museum_17ch_iem.json             # same layout in IEM AllRADecoder format
├── pyproject.toml
└── README.md
```

## Installation

Requirements: Python ≥ 3.10.

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   & ".venv\Scripts\Activate.ps1"      # PowerShell, Windows
   ```

2. Install the spatial pipeline (editable mode):
   ```
   & ".venv\Scripts\pip.exe" install -e "\CapstoneL17"
   ```

   This automatically installs `numpy`, `scipy`, `soundfile`, `sofar`, and `matplotlib`,
   which are the dependencies listed in `pyproject.toml`. **`sofar`** is the library used to read
   SOFA files containing HRTFs, and it is required by the binaural rendering module
   (`spatial_pipeline/binaural.py`): if you get `ModuleNotFoundError: No module named 'sofar'`,
   it means the active environment does not match the one where you ran `pip install -e`, or that
   the package is not yet listed among the dependencies — in that case install it manually with
   `pip install sofar`.

3. Install the dependencies for demixing (BS-RoFormer):
   ```
   pip install -r "Demixing BS-RoF\requirements.txt"
   ```
   You also need `torch`, `pyyaml`, and `ml_collections`, which are used by
   `spatial_pipeline/demix.py`.

4. (GUI only) Make sure `tkinter` is available in your Python interpreter.
   On Windows it is included in the standard installer; on Linux it may require
   `sudo apt install python3-tk`.

## Pipeline, step by step

The pipeline is divided into independent stages, each with its own script in
`src/scripts/`. Each stage reads the output of the previous stage from a default folder
inside `outputs/` (defined in `spatial_pipeline/config.py`) and can also be run on its
own, always from the project root.

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Demixing | `run_demix.py` | stereo songs in `Demixing BS-RoF/songs/` | mono stems in `outputs/demixed/` |
| 2. FOA encoding | `run_encode_foa.py` | stems in `outputs/demixed/` | FOA scene in `outputs/foa/` |
| 2. HOA encoding | `run_encode_hoa.py` | stems in `outputs/demixed/` | HOA scene (order 3) in `outputs/hoa/` |
| 3a. Decoding (museum) | `run_decode.py` | HOA scene in `outputs/hoa/` | 17-channel output in `outputs/ls17/` |
| 3b. Binaural | `run_binaural.py` | HOA scene in `outputs/hoa/` | binaural stereo in `outputs/binaural/` |
| Evaluation | `evaluate_spatial.py` | renders in `outputs/rendered/` + test cases in `outputs/test/` | plots and CSV files in `outputs/eval/` |
| Reproduction evaluation | `evaluate_pipeline_reproduction.py` | EM32 room recordings in `recs/` vs played feeds | `outputs/eval/pipeline_reproduction/pipeline_reproduction_eval.csv` |

Example, from the project root:
```
python src/scripts/run_demix.py
python src/scripts/run_encode_hoa.py
python src/scripts/run_decode.py
python src/scripts/run_binaural.py
```

The recognized stem names are `vocals`, `drums`, `bass`, `guitar`, `piano`, and `other`
(see `spatial_pipeline/scene_defaults.py`), each with a default 3D position already
designed to remain inside the museum loudspeaker dome.

## Standalone demixing

To separate a song using only BS-RoFormer, without going through the spatial pipeline:
```
bs-roformer-infer --config_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.yaml --model_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt --input_folder ./input_songs --store_dir ./outputs
```
The required dependencies are listed in the `requirements.txt` file of the BS-RoFormer
project (source: https://github.com/openmirlab/bs-roformer-infer).

## Graphical interface
<img width="1913" height="991" alt="gui_default" src="https://github.com/user-attachments/assets/e65c310f-dd59-4f86-84df-6ea2081c353d" />

```
python src/gui/gui_app.py
```
The interface provides three main functionalities. First, it enables interactive scene mapping, allowing users to position separated instrument stems within the three-dimensional sound field. A coordinate mapping utility converts two-dimensional user interactions into the corresponding azimuth and elevation parameters for the selected audio source. Second, the system supports dynamic trajectory recording, where user-defined spatial movements are sampled by the encoding engine and replayed throughout the rendered audio timeline to create continuous motion effects. Finally, the system provides binaural playback and immediate auditory feedback through an asynchronous playback mechanism that buffers and streams the processed audio. Changes made through the GUI are
immediately reflected in the spatial encoding parameters, providing immediate perceptual evaluation through headphones using the integrated binaural rendering module.


<img width="875" height="739" alt="gui_rec" src="https://github.com/user-attachments/assets/7d525ea7-1af4-42cd-8815-7377a640aa3f" />

**To sum up:**

From the GUI you can: load a song and separate it into stems ("Demix Song" button); move each stem in space (azimuth/elevation/gain, mute/solo); choose the renderer
(**Binaural**, or **Layout Speaker** to decode to a loudspeaker layout); load a custom layout (`.csv`/`.json`) or an HRTF file (`.sofa`) different from the default one; and set
the output folder and output filename. By pressing **GENERATE**, the GUI encodes the stems into HOA and writes the final result to `outputs/rendered/`.

If you choose the **Binaural** renderer, the file is always saved with the suffix `_binaural` in its name (e.g. `my_mix_binaural.wav`), even if you customize the name —
this is necessary because `evaluate_spatial.py` recognizes a file as binaural only based on that suffix.

## Measurement Setup

The system was measured and evaluated in the **Musical Acoustics Laboratory of the Violin Museum, Cremona**, on a hemispherical array of **17 Genelec 8020-series loudspeakers**.

### Coordinate system

A left-handed spherical coordinate system was adopted, centered on the listening ("sweet spot") position:

- **Azimuth (φ)**: 0°–360°, measured clockwise from the front
- **Elevation (θ)**: −90°–90°, measured upward from the horizontal plane
- **Reference height**: 0.736 m

### Procedure

1. The sweet spot was located with a laser distance meter and marked with a tripod, providing a stable reference point for the session.
2. Each loudspeaker's distance, azimuth, and elevation relative to the sweet spot were measured with the tripod-mounted laser distance meter.
3. The acoustic response of each loudspeaker was captured using **exponential sine sweep** excitation, chosen for its favorable SNR and its ability to cleanly separate the linear impulse response from harmonic distortion introduced by the transducers (which resolve at distinct time offsets after deconvolution).
4. Recordings were made at the sweet spot with an **em32 Eigenmike®** (32-capsule rigid spherical microphone array), providing the impulse response set used later for Direction-of-Arrival (DOA) validation.

### Loudspeaker layout

| Loudspeaker | Distance (m) | Azimuth (°) | Elevation (°) |
|---|---|---|---|
| A3  | 1.532 | 0   | 21.70 |
| A4  | 1.966 | 299 | 18.08 |
| A5  | 1.530 | 255 | 23.76 |
| A6  | 2.000 | 208 | 16.90 |
| A7  | 1.555 | 165 | 23.51 |
| A8  | 2.015 | 133 | 17.31 |
| A9  | 1.540 | 93  | 23.80 |
| A10 | 1.995 | 46  | 17.65 |
| A11 | 1.878 | 2   | 53.70 |
| A12 | 1.873 | 270 | 53.42 |
| A13 | 1.900 | 181 | 53.36 |
| A14 | 1.894 | 91  | 52.60 |
| A15 | 1.230 | 357 | 8.11  |
| A16 | 1.182 | 268 | 8.89  |
| A17 | 1.212 | 176 | 8.60  |
| A18 | 1.174 | 88  | 8.37  |
| A19 | 1.720 | 357 | 80.00 |

*(Loudspeaker IDs follow the lab's internal numbering; A19 is the near-zenith speaker)*

<img width="500" alt="Measured loudspeaker positions in the Musical Acoustics Laboratory, Violin Museum, Cremona" src="https://github.com/user-attachments/assets/5ab1e7d4-5409-483f-ad55-931854869d2b" />

*Spatial arrangement of the 17-loudspeaker hemispherical array, plotted from the measured azimuth/elevation/distance values above.*

<img width="850" height="478" alt="image" src="https://github.com/user-attachments/assets/ab517074-6909-440f-b6cb-88a565e129a5" />

*A 17-loudspeaker array, Musical Acoustics laboratory, Violin Museum, Cremona*


## Evaluation

```
python src/evaluation/evaluate_spatial.py
```

It reads all `.wav` files from `outputs/rendered/` (the ones generated by the GUI) **and**
the decoder test cases in `outputs/test/` (only the `ls17/` and `ls17_binaural/`
subfolders — `hoa/` holds ambisonic scenes, not speaker layouts, so it is skipped), and
produces, all under `outputs/eval/`:
- a **polar plot** for each file decoded to a loudspeaker layout (energy per loudspeaker),
  in `outputs/eval/polar/`;
- an **ITD/ILD plot** for each binaural file, plus an `itd_ild_summary.csv` summary,
  in `outputs/eval/itd_ild/`;
- if `evaluate_pipeline_reproduction.py` has been run, **re-plots** of its
  `pipeline_reproduction_eval.csv` (reproduction fidelity and angular error), in
  `outputs/eval/pipeline_reproduction/`.

You can pass a different folder with `--rendered-dir`, skip the test cases with
`--no-test`, or point at another reproduction CSV with `--pipeline-reproduction-csv`:
```
python src/evaluation/evaluate_spatial.py --rendered-dir path\to\another\folder
python src/evaluation/evaluate_spatial.py --no-test
```

## Loudspeaker layout

The museum layout (17 channels) is described in two equivalent formats:
- `measurements_transcription.csv` — the project's native format, used by default
  by `spatial_pipeline.config.MEASUREMENTS_CSV`;
- `museum_17ch_iem.json` — the same layout in IEM AllRADecoder format, generated by
  `export_iem_layout.py`, useful for importing the configuration into REAPER/AllRADecoder
  (see `AllRADecoder.rpp`).

## Main dependencies

| Library | Purpose |
|---|---|
| `numpy`, `scipy` | DSP, filters, cross-correlation |
| `soundfile` | WAV reading/writing |
| `sofar` | Reading SOFA (HRTF) files for binaural rendering |
| `matplotlib` | Evaluation plots (polar, ITD/ILD) |
| `torch`, `pyyaml`, `ml_collections` | BS-RoFormer model inference for demixing |
| `tkinter` | Graphical interface |
