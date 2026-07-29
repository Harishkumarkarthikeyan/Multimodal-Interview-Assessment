import cv2
import numpy as np
from pathlib import Path


def extract_landmarks(video_path, out_npy_path, max_frames=None, stride=1):
    """Extracts holistic landmarks (pose + face + hands) per frame and saves as .npy.

    Output shape will be (N, M, 3) where M is number of concatenated landmarks per frame.
    Missing landmarks are filled with NaN.
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise ImportError(
            "Could not import `mediapipe`. Please ensure mediapipe is installed and importable. "
            f"Original error: {e}"
        )

    if not hasattr(mp, "solutions"):
        raise ImportError(
            "Imported `mediapipe` module does not expose `solutions`. This may indicate an incompatible or broken installation."
        )

    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(str(video_path))
    landmarks = []
    frame_idx = 0
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            frame_landmarks = []
            # face
            if results.face_landmarks:
                for lm in results.face_landmarks.landmark:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 468)
            # pose
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 33)
            # left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 21)
            # right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 21)

            landmarks.append(frame_landmarks)
            frame_idx += 1
            if max_frames and len(landmarks) >= max_frames:
                break
    cap.release()
    arr = np.array(landmarks, dtype=np.float32)
    Path(out_npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, arr)
    return out_npy_path
