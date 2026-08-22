# Backend

FastAPI powers the web API and calls the multimodal processing pipeline.

## Start API

```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

## API Routes

- `GET /api/health`: server health check
- `POST /api/analyze`: upload and process an interview video
- `GET /api/jobs/{job_id}`: poll processing status
- `GET /api/demo-report`: sample report for presentation

## Pipeline

```bash
python -m backend.pipeline path\to\interview.mp4 --output-dir outputs
```

The pipeline extracts audio, transcribes speech, extracts MediaPipe landmarks,
extracts Librosa audio features, and writes a JSON assessment report.
