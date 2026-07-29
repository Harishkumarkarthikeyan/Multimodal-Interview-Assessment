import cv2
import json
from pathlib import Path
from deepface import DeepFace

def analyze_emotions(video_path, out_json_path, frame_stride=30, enforce_detection=False):
    """Sample frames from video and run DeepFace emotion analysis.

    Saves a JSON dict mapping frame_index -> analysis dict.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    results = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=enforce_detection)
            except Exception as e:
                analysis = {"error": str(e)}
            results[str(frame_idx)] = analysis
        frame_idx += 1
    cap.release()
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_json_path
