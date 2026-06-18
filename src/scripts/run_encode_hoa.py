import sys
from collections import defaultdict
from pathlib import Path

# Resolve directory structure relative to this script's location
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Add src/ to the Python path so that spatial_pipeline can be imported
# as a package without needing a formal install 
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import encode_stems_to_hoa
from spatial_pipeline.scene_defaults import STEM_TYPES, DEFAULT_POSITIONS_DEG
from spatial_pipeline.audio_io import load_mono
from spatial_pipeline.config import DEFAULT_DEMIX_DIR, DEFAULT_HOA_DIR
from spatial_pipeline.frame_processing import split_frames
from spatial_pipeline.ambisonics.core.trajectories import (
    generate_static, 
    generate_orbit,
    generate_arc_flyover,
    generate_bounce
)

def collect_stems_by_song(output_folder: Path) -> dict[str, dict[str, str]]:
    """
    Scans the demixed output folder, where each song has its own directory:
        outputs/demixed/{song_name}-stems/

    Expected stem filename format inside each song directory:
        {stem_type}-{song_name}.wav

    Returns:
        {
            "my-song": {
                "vocals": "/path/to/vocals-my-song.wav",
                "drums": "/path/to/drums-my-song.wav",
                ...
            }
        }
    """
    all_songs_stems = defaultdict(dict)

    if not output_folder.exists():
        return {}

    for song_dir in output_folder.iterdir():
        if not song_dir.is_dir():
            continue

        song_dir_name = song_dir.name

        if song_dir_name.endswith("-stems"):
            song_name = song_dir_name[:-len("-stems")]
        else:
            song_name = song_dir_name

        for wav_path in song_dir.glob("*.wav"):
            if wav_path.stem.endswith("_3d_scene"):
                continue

            matched = False
            for stem in STEM_TYPES:
                if wav_path.stem.startswith(f"{stem}-"):
                    all_songs_stems[song_name][stem] = str(wav_path)
                    matched = True
                    break

            if not matched:
                print(f"Warning: unrecognized stem file: {wav_path.name}")

    return dict(all_songs_stems)


def main():
    stems_folder = DEFAULT_DEMIX_DIR
    output_folder = DEFAULT_HOA_DIR
    output_folder.mkdir(parents=True, exist_ok=True)

    print("--- THE POSITIONING & ENCODING STAGE (HOA) ---")

    # group all stems produced by the demixing stage
    all_songs_stems = collect_stems_by_song(stems_folder)
    if not all_songs_stems:
        print(f"No stems found in {stems_folder}")
        return

    print("Assigning 3D coordinates using 3rd-Order / 16 Channels...")
    print("\nStarting Ambisonic Encoding...")

    for song_name, stem_paths in all_songs_stems.items():
        # Require all stem types to be present
        missing_stems = [stem for stem in STEM_TYPES if stem not in stem_paths]
        if missing_stems:
            print(f"Warning: '{song_name}' is missing stems {missing_stems}. Skipping.")
            continue

        # How many frames the song has by checking the first stem
        first_stem_path = list(stem_paths.values())[0]
        signal, _ = load_mono(first_stem_path)
        n_frames = len(split_frames(signal, frame_size=1024, hop_size=512))

        # Build the Trajectories dictionary
        trajectories = {}

        # =====================================================================
        # PLACEHOLDER FOR FUTURE GUI INTEGRATION
        # =====================================================================
        # TODO: Delete this if/else block when the GUI is built. 
        # The GUI should pass a configuration dictionary dictating which 
        # effect to apply to which stem. Until then, these are hardcoded 
        # to test the mathematical movement functions.
        for stem in STEM_TYPES:
            azi_deg, ele_deg = DEFAULT_POSITIONS_DEG[stem]
            
            # Assign different movement patterns based on the instrument type to create a dynamic 3D scene:
            if stem == "other":
                # Cyclone Effect: Spin the synth 5 times around the room
                trajectories[stem] = generate_orbit(n_frames, start_azi_deg=azi_deg, rotations=5.0)
                
            elif stem == "guitar":
                # Trench Run Effect: Sweep the guitar from the floor up over the listener's head
                trajectories[stem] = generate_arc_flyover(n_frames, azimuth_deg=azi_deg, start_ele_deg=-90.0, end_ele_deg=90.0)
                
            elif stem == "bass":
                # Polymath Bounce: Oscillate the bass left and right around its anchor point
                trajectories[stem] = generate_bounce(n_frames, center_azi_deg=azi_deg, width_deg=60.0, bounces=40.0)
                
            else:
                # Vocals, Drums, Piano stay perfectly still
                trajectories[stem] = generate_static(n_frames, azi_deg, ele_deg)
        # =====================================================================
        
        # Output path for the encoded HOA scene
        final_out_path = output_folder / f"{song_name}_3d_scene_hoa3.wav"
        print(f"Building HOA scene for: {song_name}")

        # Encode all stems into a single High Order Ambisonics bus
        encode_stems_to_hoa(
            stem_paths=stem_paths,
            positions_deg=DEFAULT_POSITIONS_DEG,
            trajectories=trajectories,
            out_path=str(final_out_path),
            order=3,             # 3rd Order (17 loudspeakers, 16 channels)
            normalization="sn3d" # SN3D format
        )

        print(f"Success! Saved to {final_out_path}")

    print("\n--- All HOA Processing Complete! ---")


if __name__ == "__main__":
    main()