import sys
from pathlib import Path
from collections import defaultdict

# Setup Project Paths
current_script_dir = Path(__file__).resolve().parent
src_dir = current_script_dir.parent
project_root = src_dir.parent
sys.path.append(str(src_dir))

from spatial_pipeline.pipeline import encode_stems_to_foa

def main():
    output_folder = project_root / "Demixing BS-RoF" / "outputs" 

    print("--- THE POSITIONING & ENCODING STAGE ---")
    
    # Scan the outputs folder to group stems by song
    stem_types = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    all_songs_stems = defaultdict(dict)
    
    for wav_path in output_folder.glob("*.wav"):
        # Ignore files that are already 3D scenes
        if "3d_scene" in wav_path.name:
            continue 
            
        # Figure out which stem this is and which song it belongs to
        for stem in stem_types:
            if wav_path.stem.endswith(f"_{stem}"):
                # Extract the song name
                song_name = wav_path.stem[:-(len(stem)+1)]
                all_songs_stems[song_name][stem] = str(wav_path)
                break

    if not all_songs_stems:
        print("No stems found in the outputs folder!")
        return

    # THE ROUTING ALGORITHM
    print("Assigning 3D coordinates...")
    positions_deg = {
        "vocals": (0.0, 0.0),       
        "drums": (180.0, 0.0),      
        "bass": (0.0, -20.0),       
        "guitar": (-45.0, 0.0),     
        "piano": (45.0, 0.0),       
        "other": (0.0, 60.0),       
    }

    # THE ENCODING STAGE
    print("\nStarting Ambisonic Encoding...")
    for song_name, stem_paths in all_songs_stems.items():
        if len(stem_paths) < 6:
            print(f"Warning: '{song_name}' is missing some stems. Skipping.")
            continue
            
        print(f"Building 3D Scene for: {song_name}")
        final_out_path = str(output_folder / f"{song_name}_3d_scene.wav")
        
        encode_stems_to_foa(
            stem_paths=stem_paths,
            positions_deg=positions_deg,
            out_path=final_out_path,
            convention="basic"  
        )
        print(f"Success! Saved to {final_out_path}")

    print("\n--- All Processing Complete! ---")

if __name__ == "__main__":
    main()