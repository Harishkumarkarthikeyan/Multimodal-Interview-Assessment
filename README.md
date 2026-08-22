# AI-Powered Multimodal Interview Performance Assessment

This is a full-stack final-year project for assessing interview videos with
multiple AI signals:

- Video analysis with OpenCV + MediaPipe
- Audio feature extraction with FFmpeg + Librosa
- Speech-to-text transcription with Whisper
- Explainable interview scoring and feedback with Python
- React dashboard frontend for upload, preview, scores, metrics, transcript, and report download

The app also includes a demo report button, so you can present the dashboard
even before running the heavier ML pipeline on a real video.

## Project Structure

```text
Multimodal-Interview-Assessment/
  backend/
    api.py                 FastAPI web API
    pipeline.py            Full video processing pipeline
    assessment.py          V1 scoring and feedback logic
    mediapipe_processor.py Video landmark extraction
    librosa_processor.py   Audio feature extraction
    whisper_processor.py   Speech-to-text transcription
    utils.py               FFmpeg helper
  frontend/
    index.html
    package.json
    vite.config.js
    src/
      App.jsx
      main.jsx
      styles.css
  requirements.txt
```

## Requirements

Install these first:

- Python 3.10 or newer
- Node.js 18 or newer
- FFmpeg

Check FFmpeg:

```bash
ffmpeg -version
```

If the command is not found, install FFmpeg and add it to PATH.

## 1. Backend Setup

From the project root:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

## 2. Frontend Setup

Open a second terminal and go to the frontend folder:

```bash
cd frontend
```

Install React dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

Use this mode while developing. Vite automatically sends `/api` requests to the
FastAPI backend on port `8000`.

## 3. Presentation Mode

When you want to run everything from the FastAPI server:

```bash
cd frontend
npm run build
cd ..
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

FastAPI will serve the built React app from `frontend/dist`.

## How To Use

1. Open the frontend in the browser.
2. Click `Load Demo` to show a ready-made assessment for presentation.
3. To analyze a real interview, upload a video file.
4. Click `Analyze Video`.
5. Wait for processing to finish.
6. Review the score cards, feedback, metrics, and transcript.
7. Click `Download` to save the assessment JSON.

Supported video formats:

```text
.mp4, .mov, .avi, .mkv, .webm
```

## Backend-Only Pipeline

You can also run the ML pipeline without the web app:

```bash
python -m backend.pipeline path\to\interview.mp4 --output-dir outputs
```

For `interview.mp4`, this creates:

```text
interview.wav
interview_transcript.txt
interview_landmarks.npy
interview_audio_features.npy
interview_assessment.json
```

## Important Notes

- The first real video analysis can take time because Whisper and MediaPipe are heavy.
- Whisper may download its model the first time it runs.
- `Load Demo` does not require video processing and is best for quick viva/project presentation.
- The current scoring in `backend/assessment.py` is an explainable V1 prototype. Later, you can replace it with a trained PyTorch multimodal fusion model when you have a labeled dataset.
