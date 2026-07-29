Multimodal processing pipeline

This `backend` folder contains simple modules and an orchestrator to process video files located in the repository `Data/` directory.

Pipeline steps for each video:
- Extract audio (FFmpeg required) -> `*_audio.wav`
- Transcribe audio with Whisper -> `*_transcript.txt`
- Extract MediaPipe landmarks -> `*_landmarks.npy`
- Extract audio features with Librosa -> `*_audio_features.npy`
- Analyze emotions with DeepFace -> `*_emotions.json`

Quick start

1. Install required Python packages (preferably in a virtualenv):

```bash
pip install -r requirements.txt
```

2. Ensure `ffmpeg` is installed and available on PATH.

3. Run the pipeline on the `Data/` folder (default) or a specific file:

```bash
python -m backend.pipeline --recursive
python -m backend.pipeline path/to/video.mp4
```

Notes
- The modules included are lightweight wrappers intended to be a starting point. They assume reasonable system resources.
- DeepFace and Whisper are computationally heavy; consider using smaller Whisper models or GPU acceleration where available.
