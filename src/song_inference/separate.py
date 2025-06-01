
import os
import subprocess
from pathlib import Path


def separate_vocals(audio, output_dir="/tmp/separated") -> str:
    """
    Separates vocals from the accompaniment in the given audio file using the Demucs
    tool and outputs the separated vocals to the specified directory.

    The function utilizes the Demucs CLI to perform the separation in "two-stems"
    mode, which isolates the vocals from the rest of the audio.

    Args:
        audio: The path to the audio file as a string or an instance of
            UploadedFile that supports the `read()` method.
        output_dir: The directory where the separated vocals will be stored. Defaults
            to "/tmp/separated".

    Returns:
        str: The path to the generated `vocals.wav` file.
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