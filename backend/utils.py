import subprocess
from pathlib import Path

def ensure_parent(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

def extract_audio(video_path, out_audio_path):
    """Extract audio track from video using ffmpeg (requires ffmpeg installed).

    Saves a WAV file at out_audio_path.
    """
    ensure_parent(out_audio_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_audio_path),
    ]
    subprocess.run(cmd, check=True)
