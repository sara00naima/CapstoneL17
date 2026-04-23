import sys
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
src_dir = current_script_dir.parent
project_root = src_dir.parent
sys.path.append(str(src_dir))

from spatial_pipeline.demix import demix_folder 

def main():
    input_folder = project_root / "Demixing BS-RoF" / "songs" 
    output_folder = project_root / "Demixing BS-RoF" / "outputs" 

    print("--- THE DEMIXING STAGE ---")
    
    demix_folder(str(input_folder), str(output_folder))
    
    print("\n--- Demixing Complete!---")

if __name__ == "__main__":
    main()