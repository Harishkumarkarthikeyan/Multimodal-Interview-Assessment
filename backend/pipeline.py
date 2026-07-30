import argparse
import logging
import os
from pathlib import Path
import sys

# Import local processors robustly so the file can be run as a module
# (`python -m backend.pipeline`) or directly as a script
try:
	from . import utils
	from .whisper_processor import transcribe_audio
	from .mediapipe_processor import extract_landmarks
	from .librosa_processor import extract_audio_features
except Exception:
	# Running as a script: ensure the `backend` folder (this file's dir) is on sys.path
	package_dir = Path(__file__).resolve().parent
	if str(package_dir) not in sys.path:
		sys.path.insert(0, str(package_dir))
	import utils
	from whisper_processor import transcribe_audio
	from mediapipe_processor import extract_landmarks
	from librosa_processor import extract_audio_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def find_videos(path, recursive=True):
	p = Path(path)
	if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
		return [p]
	videos = []
	if p.is_dir():
		for f in p.rglob("*") if recursive else p.iterdir():
			if f.suffix.lower() in VIDEO_EXTS:
				videos.append(f)
	return videos

def process_video(video_path, holistic_model_path=None):
	video_path = Path(video_path)
	logging.info(f"Processing video: {video_path}")
	base = video_path.with_suffix("")
	out_audio = base.with_suffix(".wav")
	transcript_path = base.parent / (base.name + "_transcript.txt")
	landmarks_path = base.parent / (base.name + "_landmarks.npy")
	audio_features_path = base.parent / (base.name + "_audio_features.npy")
	emotions_path = base.parent / (base.name + "_emotions.json")

	try:
		logging.info("Extracting audio...")
		utils.extract_audio(video_path, out_audio)
	except Exception as e:
		logging.exception("Audio extraction failed: %s", e)
		return

	try:
		logging.info("Transcribing audio with Whisper...")
		transcribe_audio(out_audio, transcript_path)
	except Exception as e:
		logging.exception("Transcription failed: %s", e)

	try:
		logging.info("Extracting landmarks with MediaPipe...")
		extract_landmarks(video_path, landmarks_path, holistic_model_path=holistic_model_path)
	except Exception as e:
		logging.exception("Landmarks extraction failed: %s", e)

	try:
		logging.info("Extracting audio features with Librosa...")
		extract_audio_features(out_audio, audio_features_path)
	except Exception as e:
		logging.exception("Audio feature extraction failed: %s", e)
	logging.info("Finished processing %s", video_path)


def main():
	parser = argparse.ArgumentParser(description="Run multimodal pipeline on videos")
	parser.add_argument("input", nargs="?", help="Video file or directory to process")
	parser.add_argument("--recursive", action="store_true", help="Recursively search directories")
	parser.add_argument(
		"--holistic-model-path",
		help="Path to a MediaPipe holistic task model file (.task or .tflite) when using MediaPipe 1.x without mp.solutions",
	)
	args = parser.parse_args()

	holistic_model_path = args.holistic_model_path or os.getenv("MEDIAPIPE_HOLISTIC_MODEL_PATH")
	if holistic_model_path:
		logging.info("Using MediaPipe holistic model path: %s", holistic_model_path)

	if args.input:
		target = Path(args.input)
	else:
		# default folder: prefer `data/train` (user-specified), fall back to `Data/Train`
		repo_root = Path(__file__).resolve().parents[1]
		candidate = repo_root / "data" / "train"
		if candidate.exists():
			target = candidate
		else:
			fallback = repo_root / "Data" / "Train"
			target = fallback if fallback.exists() else candidate

	videos = find_videos(target, recursive=args.recursive)
	if not videos:
		logging.warning("No videos found at %s", target)
		sys.exit(0)

	for v in videos:
		process_video(v, holistic_model_path=holistic_model_path)


if __name__ == "__main__":
	main()

