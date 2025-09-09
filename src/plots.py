import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from .audio_utils import AudioData, extract_features, cosine_similarity_matrix


def plot_waveform_and_spectrogram(audio: AudioData, title: Optional[str] = None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    librosa.display.waveshow(audio.y, sr=audio.sr, ax=axes[0])
    axes[0].set_title(title or os.path.basename(audio.path))
    axes[0].set_xlabel("Time (s)")

    S = librosa.feature.melspectrogram(y=audio.y, sr=audio.sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=audio.sr, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title("Mel Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    return fig, axes


def plot_similarity_matrix(F_rec: np.ndarray, F_samp: np.ndarray, title: str = "Cosine Similarity"):
    S = cosine_similarity_matrix(F_rec, F_samp)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(S, origin='lower', aspect='auto', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel('Sample frames')
    ax.set_ylabel('Recording frames')
    fig.colorbar(im, ax=ax)
    return fig, ax
