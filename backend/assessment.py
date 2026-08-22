import json
import re
from pathlib import Path

import numpy as np


FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "actually",
    "basically",
    "literally",
    "you know",
    "i mean",
}


def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def _score_range(value, ideal_low, ideal_high, hard_low, hard_high):
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return _clamp(100.0 * (value - hard_low) / max(ideal_low - hard_low, 1e-6))
    return _clamp(100.0 * (hard_high - value) / max(hard_high - ideal_high, 1e-6))


def _load_audio_features(path):
    data = np.load(path, allow_pickle=True).item()
    return data


def _word_count(text):
    return len(re.findall(r"\b[\w']+\b", text.lower()))


def _sentence_count(text):
    matches = re.findall(r"[.!?]+", text)
    return max(1, len(matches))


def _filler_count(text):
    lowered = text.lower()
    total = 0
    for filler in FILLER_WORDS:
        total += len(re.findall(rf"\b{re.escape(filler)}\b", lowered))
    return total


def _analyze_landmarks(landmarks_path):
    landmarks = np.load(landmarks_path)
    if landmarks.size == 0:
        return {
            "sampled_frames": 0,
            "face_visibility": 0.0,
            "pose_visibility": 0.0,
            "movement_stability": 0.0,
        }

    face = landmarks[:, :468, :]
    pose = landmarks[:, 468:501, :]

    face_visibility = float(np.mean(~np.isnan(face[:, :, 0])))
    pose_visibility = float(np.mean(~np.isnan(pose[:, :, 0])))

    nose = pose[:, 0, :2]
    valid_nose = nose[~np.isnan(nose).any(axis=1)]
    if len(valid_nose) > 2:
        movement = float(np.mean(np.linalg.norm(np.diff(valid_nose, axis=0), axis=1)))
        movement_stability = _clamp(100.0 - movement * 700.0)
    else:
        movement_stability = 35.0 if pose_visibility > 0 else 0.0

    return {
        "sampled_frames": int(landmarks.shape[0]),
        "face_visibility": face_visibility,
        "pose_visibility": pose_visibility,
        "movement_stability": movement_stability,
    }


def build_assessment(video_path, transcript_path, audio_features_path, landmarks_path, out_json_path):
    """Create a human-readable V1 interview assessment report."""
    transcript = Path(transcript_path).read_text(encoding="utf-8").strip()
    audio = _load_audio_features(audio_features_path)
    visual = _analyze_landmarks(landmarks_path)

    words = _word_count(transcript)
    sentences = _sentence_count(transcript)
    duration_seconds = max(float(audio.get("duration_seconds", 0.0)), 1.0)
    words_per_minute = words / duration_seconds * 60.0
    fillers = _filler_count(transcript)
    filler_rate = fillers / max(words, 1)
    avg_sentence_length = words / sentences

    pace_score = _score_range(words_per_minute, 110, 155, 55, 220)
    filler_score = _clamp(100.0 - filler_rate * 700.0)
    answer_depth_score = _score_range(words, 55, 180, 10, 320)
    clarity_score = _score_range(avg_sentence_length, 8, 24, 3, 42)

    voice_energy = float(audio.get("mean_rms", 0.0))
    energy_score = _score_range(voice_energy, 0.025, 0.12, 0.003, 0.25)
    steadiness_score = _clamp(100.0 - float(audio.get("std_rms", 0.0)) * 400.0)

    face_score = visual["face_visibility"] * 100.0
    posture_score = visual["pose_visibility"] * 100.0
    movement_score = visual["movement_stability"]

    communication = _clamp(
        pace_score * 0.30
        + filler_score * 0.25
        + answer_depth_score * 0.25
        + clarity_score * 0.20
    )
    confidence = _clamp(
        energy_score * 0.35
        + steadiness_score * 0.20
        + face_score * 0.25
        + movement_score * 0.20
    )
    non_verbal = _clamp(face_score * 0.45 + posture_score * 0.25 + movement_score * 0.30)
    employability = _clamp(communication * 0.45 + confidence * 0.35 + non_verbal * 0.20)

    strengths = []
    improvements = []

    if pace_score >= 75:
        strengths.append("Speaking pace is close to a natural interview range.")
    else:
        improvements.append("Practice answering at a steady pace of roughly 110-155 words per minute.")

    if filler_score >= 80:
        strengths.append("Low filler-word usage keeps the answer clean.")
    else:
        improvements.append("Reduce filler words by pausing briefly before continuing.")

    if answer_depth_score >= 75:
        strengths.append("Answer length provides enough detail for evaluation.")
    else:
        improvements.append("Add more concrete examples using Situation, Task, Action, and Result.")

    if face_score >= 70:
        strengths.append("Face visibility is strong, which supports visual assessment.")
    else:
        improvements.append("Keep your face centered and well lit throughout the recording.")

    if movement_score < 60:
        improvements.append("Reduce unnecessary head movement and maintain a composed posture.")

    if not transcript:
        improvements.append("No transcript was detected, so content-based scoring is limited.")

    report = {
        "video": str(video_path),
        "scores": {
            "confidence": round(confidence, 2),
            "communication": round(communication, 2),
            "non_verbal_presence": round(non_verbal, 2),
            "employability": round(employability, 2),
        },
        "metrics": {
            "duration_seconds": round(duration_seconds, 2),
            "word_count": words,
            "words_per_minute": round(words_per_minute, 2),
            "filler_words": fillers,
            "average_sentence_length": round(avg_sentence_length, 2),
            "voice_energy": round(voice_energy, 5),
            **visual,
        },
        "strengths": strengths[:4],
        "improvements": improvements[:5],
        "transcript": transcript,
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
