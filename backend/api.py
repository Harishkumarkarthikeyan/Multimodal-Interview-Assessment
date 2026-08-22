import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="Multimodal Interview Assessment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
jobs_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _set_job(job_id, **updates):
    with jobs_lock:
        current = jobs.setdefault(job_id, {})
        current.update(updates)
        current["updated_at"] = _now()
        return dict(current)


def _get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def _safe_filename(filename):
    name = Path(filename or "interview.mp4").name
    return "".join(char for char in name if char.isalnum() or char in "._- ")[:120]


def _run_pipeline(job_id, video_path):
    _set_job(job_id, status="processing", message="Analyzing video, audio, and transcript")
    try:
        from .pipeline import process_video

        job_output_dir = OUTPUT_DIR / job_id
        report = process_video(
            video_path,
            output_dir=job_output_dir,
            whisper_model="base",
            language="en",
            landmark_stride=3,
        )
        _set_job(
            job_id,
            status="complete",
            message="Assessment complete",
            report=report,
            report_path=str(job_output_dir / f"{Path(video_path).stem}_assessment.json"),
        )
    except Exception as exc:
        _set_job(job_id, status="failed", message=str(exc))


@app.get("/")
def index():
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": (
            "React build not found. Run `npm install` and `npm run build` "
            "inside the frontend folder, or use `npm run dev` for frontend development."
        )
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Interview assessment server is running"}


@app.post("/api/analyze")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="Upload a video file: mp4, mov, avi, mkv, or webm")

    job_id = uuid.uuid4().hex
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = _safe_filename(file.filename)
    video_path = UPLOAD_DIR / f"{job_id}_{filename}"

    with video_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    _set_job(
        job_id,
        status="queued",
        message="Video uploaded",
        filename=filename,
        video_path=str(video_path),
        created_at=_now(),
    )
    background_tasks.add_task(_run_pipeline, job_id, video_path)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/demo-report")
def demo_report():
    return {
        "video": "demo_interview.mp4",
        "scores": {
            "confidence": 82.4,
            "communication": 78.6,
            "non_verbal_presence": 84.1,
            "employability": 81.2,
        },
        "metrics": {
            "duration_seconds": 96.3,
            "word_count": 214,
            "words_per_minute": 133.3,
            "filler_words": 6,
            "average_sentence_length": 17.8,
            "voice_energy": 0.05241,
            "sampled_frames": 770,
            "face_visibility": 0.91,
            "pose_visibility": 0.83,
            "movement_stability": 78.6,
        },
        "strengths": [
            "Speaking pace is close to a natural interview range.",
            "Answer length provides enough detail for evaluation.",
            "Face visibility is strong, which supports visual assessment.",
            "Voice energy is steady and suitable for an interview setting.",
        ],
        "improvements": [
            "Reduce filler words by pausing briefly before continuing.",
            "Add more measurable outcomes when describing project experience.",
            "Keep your posture consistent during longer answers.",
        ],
        "transcript": (
            "I am a final year computer science student with experience in Python, "
            "machine learning, and web development. In my recent project, I built a "
            "multimodal interview assessment system that analyzes speech, video, and "
            "language quality to generate useful feedback for candidates."
        ),
    }


if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
