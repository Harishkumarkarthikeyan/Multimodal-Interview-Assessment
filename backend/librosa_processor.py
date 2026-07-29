import numpy as np
import librosa
from pathlib import Path

def extract_audio_features(audio_path, out_npy_path, sr=16000, n_mfcc=13):
    """Extract common audio features (MFCCs, chroma, mel) and save as .npy."""
    y, sr_res = librosa.load(str(audio_path), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_res, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr_res)
    mel = librosa.feature.melspectrogram(y=y, sr=sr_res)
    features = {
        "mfcc": mfcc,
        "chroma": chroma,
        "mel": mel,
        "sr": sr_res,
    }
    Path(out_npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, features, allow_pickle=True)
    return out_npy_path
