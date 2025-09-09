from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
from scipy.spatial.distance import cdist


@dataclass
class AudioData:
    y: np.ndarray
    sr: int
    path: str


def load_audio(path: str, target_sr: Optional[int] = 16000, mono: bool = True) -> AudioData:
    """
    Load audio file using librosa/soundfile.
    - Converts to mono by default.
    - Resamples to target_sr if provided.
    """
    y, sr = librosa.load(path, sr=None, mono=mono)
    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return AudioData(y=y, sr=sr, path=path)


def trim_silence(audio: AudioData, top_db: float = 30.0) -> AudioData:
    """Trim leading and trailing silence using energy-based detection."""
    intervals = librosa.effects.split(audio.y, top_db=top_db)
    if len(intervals) == 0:
        return audio
    y_trimmed = np.concatenate([audio.y[s:e] for s, e in intervals])
    return AudioData(y=y_trimmed, sr=audio.sr, path=audio.path)


def extract_features(audio: AudioData, n_mfcc: int = 13, hop_length: int = 160, n_fft: int = 400,
                     add_deltas: bool = True, add_prosody: bool = True) -> np.ndarray:
    """
    Extract MFCCs + optional deltas + optional prosody/spectral features.
    Returns features as (T, D) matrix.
    """
    y = librosa.effects.preemphasis(audio.y)

    # Spectrogram & Mel
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=audio.sr, n_mels=40)

    # MFCCs
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)

    feats = [mfcc]

    if add_deltas:
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        feats += [d1, d2]

    if add_prosody:
        # Pitch (f0) using yin
        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=audio.sr, frame_length=n_fft, hop_length=hop_length)
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            f0 = np.zeros(mfcc.shape[1])
        # Energy
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        # Spectral features
        centroid = librosa.feature.spectral_centroid(S=S, sr=audio.sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=audio.sr)[0]
        contrast = librosa.feature.spectral_contrast(S=S, sr=audio.sr)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=audio.sr)[0]

        prosody = np.vstack([
            f0,
            rms,
            centroid,
            bandwidth,
            rolloff,
            contrast
        ])
        feats.append(prosody)

    F = np.vstack(feats)  # (D, T)
    return F.T  # (T, D)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two (T, D) matrices."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return np.dot(A_norm, B_norm.T)


def dtw_distance(A: np.ndarray, B: np.ndarray, metric: str = 'cosine') -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute DTW distance between sequences A and B using librosa.sequence.dtw.
    A and B are (T, D) feature matrices.
    metric can be 'cosine' or any metric supported by scipy.spatial.distance.cdist.
    Returns (accumulated distance, path dict with index arrays).
    """
    if metric == 'cosine':
        # DTW expects a cost matrix; cosine distance = 1 - cosine_similarity
        cost = 1.0 - cosine_similarity_matrix(A, B)
    else:
        cost = cdist(A, B, metric=metric)

    # librosa.sequence.dtw expects a cost matrix via C=...
    acc_cost, wp = librosa.sequence.dtw(C=cost)
    dist = float(acc_cost[-1, -1])
    # wp is a list of (i, j) pairs from end to start; reverse to start->end and split
    wp = wp[::-1]
    index1 = np.array([i for (i, _j) in wp], dtype=int)
    index2 = np.array([j for (_i, j) in wp], dtype=int)
    return dist, {"index1": index1, "index2": index2}


def compare_recording_to_samples(recording_path: str, samples: List[str], target_sr: int = 16000) -> List[Dict]:
    """
    For a given recording, compute similarity to each sample file.
    Returns a list of dicts with distances and summary statistics.
    """
    rec = trim_silence(load_audio(recording_path, target_sr=target_sr))
    F_rec = zscore(extract_features(rec))

    results = []
    for spath in samples:
        samp = trim_silence(load_audio(spath, target_sr=target_sr))
        F_samp = zscore(extract_features(samp))

        # DTW distance (lower is better)
        d_dtw, path = dtw_distance(F_rec, F_samp, metric='cosine')

        # Framewise max similarity summary
        S = cosine_similarity_matrix(F_rec, F_samp)
        framewise_best = S.max(axis=1)
        sim_mean = float(framewise_best.mean())
        sim_p90 = float(np.percentile(framewise_best, 90))

        results.append({
            "sample": spath,
            "dtw_distance": d_dtw,
            "sim_mean": sim_mean,
            "sim_p90": sim_p90,
            "n_frames_rec": int(F_rec.shape[0]),
            "n_frames_sample": int(F_samp.shape[0])
        })
    # Sort by DTW distance ascending, then by sim_mean desc
    results.sort(key=lambda x: (x["dtw_distance"], -x["sim_mean"]))
    return results
