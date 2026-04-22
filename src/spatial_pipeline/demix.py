from pathlib import Path

def demix_with_external_tool(input_wav: str, output_dir: str) -> dict:
    """
    Placeholder:
    runs BS-RoFormer/Broformer externally and returns the paths of the separated stems.
    """
    out = Path(output_dir)
    return {
        "vocals": str(out / "vocals.wav"),
        "drums": str(out / "drums.wav"),
        "bass": str(out / "bass.wav"),
        "other": str(out / "other.wav"),
    }