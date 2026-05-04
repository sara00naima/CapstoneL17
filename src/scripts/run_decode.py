import sys
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_pipeline.pipeline import decode_scene_for_ls17

def main():
    print("--- SPATIAL AUDIO DECODING STAGE ---")
    
    output_folder = PROJECT_ROOT / "Demixing BS-RoF" / "outputs"
    scene_files = list(output_folder.glob("*_3d_scene_hoa3.wav"))
    
    if not scene_files:
        print("No HOA3 scene files found. Please execute the encoding stage prior to decoding.")
        return

    for scene_path in scene_files:
        song_name = scene_path.stem.replace("_3d_scene_hoa3", "")
        print(f"\nDecoding 3D Scene: {song_name}")
        
        final_out_path = output_folder / f"{song_name}_17ch_museum_mix.wav"
        
        num_speakers = decode_scene_for_ls17(str(scene_path), str(final_out_path), order=3)
        
        print(f"Decoding successful! Generated {num_speakers} discrete speaker feeds.")
        print(f"Saved to: {final_out_path}")

    print("\n--- All Processing Complete! ---")

if __name__ == "__main__":
    main()