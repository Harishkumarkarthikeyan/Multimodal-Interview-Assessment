import argparse
import logging
import sys
from pathlib import Path

try:
    from . import utils
    from .assessment import build_assessment
    from .librosa_processor import extract_audio_features
    from .mediapipe_processor import extract_landmarks
    from .whisper_processor import transcribe_audio
except ImportError:
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

    import utils
    from assessment import build_assessment
    from librosa_processor import extract_audio_features
    from mediapipe_processor import extract_landmarks
    from whisper_processor import transcribe_audio


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def find_videos(path, recursive=True):
    target = Path(path)
    if target.is_file() and target.suffix.lower() in VIDEO_EXTS:
        return [target]

    if not target.is_dir():
        return []

    iterator = target.rglob("*") if recursive else target.iterdir()
    return sorted(file for file in iterator if file.suffix.lower() in VIDEO_EXTS)


def output_paths(video_path, output_dir=None):
    video_path = Path(video_path)
    base_dir = Path(output_dir) if output_dir else video_path.parent
    base_name = video_path.stem
    return {
        "audio": base_dir / f"{base_name}.wav",
        "transcript": base_dir / f"{base_name}_transcript.txt",
        "landmarks": base_dir / f"{base_name}_landmarks.npy",
        "audio_features": base_dir / f"{base_name}_audio_features.npy",
        "assessment": base_dir / f"{base_name}_assessment.json",
    }


def process_video(
    video_path,
    output_dir=None,
    whisper_model="base",
    language=None,
    landmark_stride=2,
    max_frames=None,
):
    video_path = Path(video_path)
    paths = output_paths(video_path, output_dir=output_dir)

    logging.info("Processing video: %s", video_path)

    logging.info("Extracting audio")
    utils.extract_audio(video_path, paths["audio"])

    logging.info("Transcribing audio with Whisper model '%s'", whisper_model)
    transcribe_audio(
        paths["audio"],
        paths["transcript"],
        model_name=whisper_model,
        language=language,
    )

    logging.info("Extracting MediaPipe landmarks")
    extract_landmarks(
        video_path,
        paths["landmarks"],
        stride=landmark_stride,
        max_frames=max_frames,
    )

    logging.info("Extracting Librosa audio features")
    extract_audio_features(paths["audio"], paths["audio_features"])

    logging.info("Building assessment report")
    report = build_assessment(
        video_path,
        paths["transcript"],
        paths["audio_features"],
        paths["landmarks"],
        paths["assessment"],
    )

    logging.info("Assessment saved to %s", paths["assessment"])
    return report


def default_input_path():
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (
        repo_root / "data" / "train",
        repo_root / "data",
        repo_root / "Data" / "Train",
        repo_root / "Data",
    ):
        if candidate.exists():
            return candidate
    return repo_root / "data"


def main():
    parser = argparse.ArgumentParser(description="Run multimodal interview assessment.")
    parser.add_argument("input", nargs="?", help="Video file or directory to process")
    parser.add_argument("--output-dir", help="Directory for generated outputs")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name")
    parser.add_argument("--language", help="Optional Whisper language code, for example en")
    parser.add_argument("--landmark-stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, help="Limit sampled frames per video")
    args = parser.parse_args()

    target = Path(args.input) if args.input else default_input_path()
    videos = find_videos(target, recursive=args.recursive)

    if not videos:
        logging.warning("No videos found at %s", target)
        return 0

    for video in videos:
        try:
            process_video(
                video,
                output_dir=args.output_dir,
                whisper_model=args.whisper_model,
                language=args.language,
                landmark_stride=max(1, args.landmark_stride),
                max_frames=args.max_frames,
            )
        except Exception:
            logging.exception("Failed to process %s", video)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
