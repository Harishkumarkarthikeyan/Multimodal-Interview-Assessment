import shutil
import subprocess
from pathlib import Path


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path, out_audio_path):
    """Extract a mono 16 kHz WAV audio track from a video file."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required but was not found on PATH.")

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
    return out_audio_path
