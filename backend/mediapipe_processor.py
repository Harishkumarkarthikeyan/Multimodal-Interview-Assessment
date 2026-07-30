import cv2
import numpy as np
from pathlib import Path


def extract_landmarks(video_path, out_npy_path, max_frames=None, stride=1, holistic_model_path=None):
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

    use_tasks_api = False
    if hasattr(mp, "solutions"):
        mp_holistic = mp.solutions.holistic
        holistic_context = mp_holistic.Holistic(static_image_mode=False)
    else:
        task_import_errors = []
        HolisticLandmarker = None
        mp_image_module = None

        for vision_path in (
            "mediapipe.tasks.python.vision",
            "mediapipe.tasks.vision",
        ):
            try:
                module = __import__(vision_path, fromlist=["HolisticLandmarker"])
                HolisticLandmarker = getattr(module, "HolisticLandmarker", None)
                mp_image_module = __import__(vision_path + ".core.image", fromlist=["image"])
                if HolisticLandmarker is not None:
                    break
            except Exception as e:
                task_import_errors.append(f"{vision_path}: {e}")
                HolisticLandmarker = None
                mp_image_module = None

        if HolisticLandmarker is None or mp_image_module is None:
            raise ImportError(
                "Imported `mediapipe` does not expose `mp.solutions` and the MediaPipe Tasks API import failed. "
                "Please install a compatible mediapipe package with task support. "
                f"Original errors: {task_import_errors}"
            )

        if holistic_model_path is None:
            raise FileNotFoundError(
                "The installed mediapipe package does not expose `mp.solutions`. "
                "Please install a mediapipe version that includes `mp.solutions`, or pass a valid task model file path via `holistic_model_path`."
            )

        holistic_model_path = Path(holistic_model_path)
        if not holistic_model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe holistic task model not found at: {holistic_model_path}"
            )

        holistic_context = HolisticLandmarker.create_from_model_path(str(holistic_model_path))
        use_tasks_api = True

    cap = cv2.VideoCapture(str(video_path))
    landmarks = []
    frame_idx = 0

    if use_tasks_api:
        def to_mp_image(cv_image):
            return mp_image_module.Image(image_format=mp_image_module.ImageFormat.SRGB, data=cv_image)

    def iter_landmarks(landmarks_obj):
        if landmarks_obj is None:
            return []
        if hasattr(landmarks_obj, "landmark"):
            return landmarks_obj.landmark
        return landmarks_obj

    with holistic_context as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_tasks_api:
                mp_image = to_mp_image(image)
                results = holistic.detect(mp_image)
            else:
                results = holistic.process(image)
            frame_landmarks = []
            # face
            face_landmarks = iter_landmarks(results.face_landmarks)
            if face_landmarks:
                for lm in face_landmarks:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 468)
            # pose
            pose_landmarks = iter_landmarks(results.pose_landmarks)
            if pose_landmarks:
                for lm in pose_landmarks:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 33)
            # left hand
            left_hand_landmarks = iter_landmarks(results.left_hand_landmarks)
            if left_hand_landmarks:
                for lm in left_hand_landmarks:
                    frame_landmarks.append((lm.x, lm.y, lm.z))
            else:
                frame_landmarks.extend([(np.nan, np.nan, np.nan)] * 21)
            # right hand
            right_hand_landmarks = iter_landmarks(results.right_hand_landmarks)
            if right_hand_landmarks:
                for lm in right_hand_landmarks:
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
