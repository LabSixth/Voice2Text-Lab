import os
import subprocess
from pathlib import Path


def separate_vocals(audio, output_dir="/tmp/separated") -> str:
    """Separate vocals from the audio file using Demucs.
    Args:
        audio (str or UploadedFile): Path to the audio file or an UploadedFile object.
        output_dir (str): Directory to save the separated vocals.
    Returns:
        str: Path to the separated vocals file.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    # Save UploadedFile if needed
    if hasattr(audio, 'read'):
        tmp_path = f"/tmp/{audio.name}"
        with open(tmp_path, 'wb') as f:
            f.write(audio.read())
        audio_path = tmp_path
    else:
        audio_path = audio

    # Call Demucs CLI to separate vocals (two stems mode)
    # --two-stems vocals: outputs vocals and accompaniment
    cmd = [
        "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", output_dir,
        audio_path
    ]
    subprocess.run(cmd, check=True)

    # Demucs outputs into {output_dir}/htdemucs/{track_name}/vocals.wav
    basename = Path(audio_path).stem
    out_dir = Path(output_dir) / "htdemucs" / basename
    vocals_path = out_dir / "vocals.wav"
    return str(vocals_path)