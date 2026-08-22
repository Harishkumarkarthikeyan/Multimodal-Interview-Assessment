from pathlib import Path

import librosa
import numpy as np


def extract_audio_features(audio_path, out_npy_path, sr=16000, n_mfcc=13):
    """Extract audio features and save them as a NumPy dictionary."""
    y, sample_rate = librosa.load(str(audio_path), sr=sr, mono=True)

    if y.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    tempo = librosa.feature.tempo(y=y, sr=sample_rate)

    features = {
        "mfcc": mfcc,
        "chroma": chroma,
        "mel": mel,
        "rms": rms,
        "zero_crossing_rate": zcr,
        "sample_rate": sample_rate,
        "duration_seconds": float(librosa.get_duration(y=y, sr=sample_rate)),
        "mean_rms": float(np.mean(rms)),
        "std_rms": float(np.std(rms)),
        "mean_zero_crossing_rate": float(np.mean(zcr)),
        "tempo_bpm": float(tempo[0]) if len(tempo) else 0.0,
    }

    Path(out_npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, features, allow_pickle=True)
    return out_npy_path
