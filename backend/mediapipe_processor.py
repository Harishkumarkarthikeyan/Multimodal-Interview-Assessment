from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


FACE_POINTS = 468
POSE_POINTS = 33
HAND_POINTS = 21
TOTAL_POINTS = FACE_POINTS + POSE_POINTS + HAND_POINTS + HAND_POINTS


def _empty_points(count):
    return [(np.nan, np.nan, np.nan)] * count


def _append_landmarks(frame_landmarks, landmarks_obj, expected_count):
    if landmarks_obj is None:
        frame_landmarks.extend(_empty_points(expected_count))
        return

    detected = landmarks_obj.landmark[:expected_count]
    for landmark in detected:
        frame_landmarks.append((landmark.x, landmark.y, landmark.z))

    if len(detected) < expected_count:
        frame_landmarks.extend(_empty_points(expected_count - len(detected)))


def extract_landmarks(video_path, out_npy_path, max_frames=None, stride=2):
    """Extract face, pose, and hand landmarks from a video.

    The saved array has shape (num_sampled_frames, 543, 3). Missing landmarks are
    stored as NaN so downstream scoring can tell the difference between a real
    zero coordinate and an undetected point.
    """
    if not hasattr(mp, "solutions"):
        raise ImportError(
            "This project expects the classic MediaPipe Solutions API. "
            "Install a compatible mediapipe version, for example mediapipe==0.10.21."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    all_landmarks = []
    frame_index = 0
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
    )

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            frame_landmarks = []
            _append_landmarks(frame_landmarks, results.face_landmarks, FACE_POINTS)
            _append_landmarks(frame_landmarks, results.pose_landmarks, POSE_POINTS)
            _append_landmarks(frame_landmarks, results.left_hand_landmarks, HAND_POINTS)
            _append_landmarks(frame_landmarks, results.right_hand_landmarks, HAND_POINTS)

            if len(frame_landmarks) != TOTAL_POINTS:
                raise ValueError(
                    f"Expected {TOTAL_POINTS} landmarks, got {len(frame_landmarks)}"
                )

            all_landmarks.append(frame_landmarks)
            frame_index += 1

            if max_frames is not None and len(all_landmarks) >= max_frames:
                break
    finally:
        cap.release()
        holistic.close()

    landmarks_array = np.array(all_landmarks, dtype=np.float32)
    Path(out_npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, landmarks_array)
    return out_npy_path
