#!/usr/bin/env python3
"""
Detailed phonetic analysis of manu pronunciation
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# Add src to path
sys.path.append('src')
from audio_utils import load_audio, trim_silence

def detailed_phonetic_analysis():
    """Detailed phonetic analysis with spectrograms"""
    
    # File paths
    recording_path = "/home/nk21137/OneDrive/5years/graduation_research/ainu/data/recordings/yabuki/manu.wav"
    sample_path = "/home/nk21137/OneDrive/5years/graduation_research/ainu/data/samples/vacabulary/manu.wav"
    
    print("🔬 Detailed Phonetic Analysis: 'manu' (bear)")
    print("=" * 50)
    
    # Load and process audio
    TARGET_SR = 16000
    TRIM_DB = 30.0
    
    rec_audio = load_audio(recording_path, target_sr=TARGET_SR, mono=True)
    samp_audio = load_audio(sample_path, target_sr=TARGET_SR, mono=True)
    
    rec_trimmed = trim_silence(rec_audio, top_db=TRIM_DB)
    samp_trimmed = trim_silence(samp_audio, top_db=TRIM_DB)
    
    # Create detailed spectrograms
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Phonetic Analysis: 'manu' Pronunciation Comparison", fontsize=16)
    
    # Time vectors
    rec_time = np.linspace(0, len(rec_trimmed.y) / TARGET_SR, len(rec_trimmed.y))
    samp_time = np.linspace(0, len(samp_trimmed.y) / TARGET_SR, len(samp_trimmed.y))
    
    # Row 1: Waveforms
    axes[0,0].plot(rec_time, rec_trimmed.y, 'b-', alpha=0.8)
    axes[0,0].set_title('Recording: Yabuki "manu"')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(samp_time, samp_trimmed.y, 'r-', alpha=0.8)
    axes[0,1].set_title('Sample: Reference "manu"')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].grid(True, alpha=0.3)
    
    # Row 2: Mel spectrograms
    rec_mel = librosa.feature.melspectrogram(y=rec_trimmed.y, sr=TARGET_SR, 
                                           hop_length=160, n_fft=400, n_mels=64)
    rec_mel_db = librosa.power_to_db(rec_mel, ref=np.max)
    
    samp_mel = librosa.feature.melspectrogram(y=samp_trimmed.y, sr=TARGET_SR,
                                            hop_length=160, n_fft=400, n_mels=64)
    samp_mel_db = librosa.power_to_db(samp_mel, ref=np.max)
    
    img1 = librosa.display.specshow(rec_mel_db, sr=TARGET_SR, hop_length=160,
                                   x_axis='time', y_axis='mel', ax=axes[1,0])
    axes[1,0].set_title('Recording: Mel Spectrogram')
    plt.colorbar(img1, ax=axes[1,0], format='%+2.0f dB')
    
    img2 = librosa.display.specshow(samp_mel_db, sr=TARGET_SR, hop_length=160,
                                   x_axis='time', y_axis='mel', ax=axes[1,1])
    axes[1,1].set_title('Sample: Mel Spectrogram')
    plt.colorbar(img2, ax=axes[1,1], format='%+2.0f dB')
    
    # Row 3: MFCC comparison
    rec_mfcc = librosa.feature.mfcc(y=rec_trimmed.y, sr=TARGET_SR, n_mfcc=13,
                                   hop_length=160, n_fft=400)
    samp_mfcc = librosa.feature.mfcc(y=samp_trimmed.y, sr=TARGET_SR, n_mfcc=13,
                                    hop_length=160, n_fft=400)
    
    img3 = librosa.display.specshow(rec_mfcc, sr=TARGET_SR, hop_length=160,
                                   x_axis='time', ax=axes[2,0])
    axes[2,0].set_title('Recording: MFCC Features')
    axes[2,0].set_ylabel('MFCC')
    plt.colorbar(img3, ax=axes[2,0])
    
    img4 = librosa.display.specshow(samp_mfcc, sr=TARGET_SR, hop_length=160,
                                   x_axis='time', ax=axes[2,1])
    axes[2,1].set_title('Sample: MFCC Features')
    axes[2,1].set_ylabel('MFCC')
    plt.colorbar(img4, ax=axes[2,1])
    
    plt.tight_layout()
    
    # Save detailed analysis
    output_dir = Path('outputs/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_path = output_dir / 'manu_detailed_phonetic_analysis.png'
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
    print(f"📊 Detailed analysis saved: {detailed_path}")
    
    # Audio characteristics analysis
    print("\n🎵 AUDIO CHARACTERISTICS")
    print("=" * 30)
    
    # Duration analysis
    rec_dur = len(rec_trimmed.y) / TARGET_SR
    samp_dur = len(samp_trimmed.y) / TARGET_SR
    duration_diff = abs(rec_dur - samp_dur)
    
    print(f"Recording duration: {rec_dur:.3f}s")
    print(f"Sample duration:    {samp_dur:.3f}s")
    print(f"Duration difference: {duration_diff:.3f}s ({duration_diff/samp_dur*100:.1f}%)")
    
    # Energy analysis
    rec_energy = np.mean(rec_trimmed.y**2)
    samp_energy = np.mean(samp_trimmed.y**2)
    energy_ratio = rec_energy / samp_energy
    
    print(f"\nEnergy levels:")
    print(f"Recording RMS energy: {np.sqrt(rec_energy):.4f}")
    print(f"Sample RMS energy:    {np.sqrt(samp_energy):.4f}")
    print(f"Energy ratio:         {energy_ratio:.2f}")
    
    # Spectral characteristics
    rec_centroid = np.mean(librosa.feature.spectral_centroid(y=rec_trimmed.y, sr=TARGET_SR))
    samp_centroid = np.mean(librosa.feature.spectral_centroid(y=samp_trimmed.y, sr=TARGET_SR))
    
    print(f"\nSpectral characteristics:")
    print(f"Recording spectral centroid: {rec_centroid:.1f} Hz")
    print(f"Sample spectral centroid:    {samp_centroid:.1f} Hz")
    print(f"Centroid difference:         {abs(rec_centroid - samp_centroid):.1f} Hz")
    
    # MFCC correlation
    # Align MFCC lengths for correlation
    min_frames = min(rec_mfcc.shape[1], samp_mfcc.shape[1])
    rec_mfcc_aligned = rec_mfcc[:, :min_frames]
    samp_mfcc_aligned = samp_mfcc[:, :min_frames]
    
    mfcc_correlations = []
    for i in range(min(13, rec_mfcc_aligned.shape[0])):
        corr = np.corrcoef(rec_mfcc_aligned[i], samp_mfcc_aligned[i])[0,1]
        mfcc_correlations.append(corr)
    
    mean_mfcc_corr = np.mean([c for c in mfcc_correlations if not np.isnan(c)])
    
    print(f"\nMFCC correlation:")
    print(f"Mean MFCC correlation: {mean_mfcc_corr:.3f}")
    
    # Overall assessment
    print(f"\n🎯 PHONETIC ASSESSMENT")
    print("=" * 25)
    
    similarity_factors = []
    
    # Duration similarity (closer to 1.0 is better)
    duration_sim = 1.0 - min(duration_diff / max(rec_dur, samp_dur), 1.0)
    similarity_factors.append(("Duration", duration_sim))
    
    # Energy similarity
    energy_sim = 1.0 - min(abs(energy_ratio - 1.0), 1.0)
    similarity_factors.append(("Energy", energy_sim))
    
    # Spectral similarity
    spectral_sim = 1.0 - min(abs(rec_centroid - samp_centroid) / max(rec_centroid, samp_centroid), 1.0)
    similarity_factors.append(("Spectral", spectral_sim))
    
    # MFCC similarity
    mfcc_sim = max(0, mean_mfcc_corr)
    similarity_factors.append(("MFCC", mfcc_sim))
    
    for factor_name, score in similarity_factors:
        print(f"{factor_name:12}: {score:.3f}")
    
    overall_similarity = np.mean([score for _, score in similarity_factors])
    print(f"{'Overall':12}: {overall_similarity:.3f}")
    
    if overall_similarity > 0.8:
        assessment = "EXCELLENT pronunciation match"
    elif overall_similarity > 0.6:
        assessment = "GOOD pronunciation match"
    elif overall_similarity > 0.4:
        assessment = "MODERATE pronunciation similarity"
    else:
        assessment = "LOW pronunciation similarity"
    
    print(f"\n📝 Final Assessment: {assessment}")
    
    return {
        'recording_duration': rec_dur,
        'sample_duration': samp_dur,
        'duration_similarity': duration_sim,
        'energy_similarity': energy_sim,
        'spectral_similarity': spectral_sim,
        'mfcc_similarity': mfcc_sim,
        'overall_similarity': overall_similarity,
        'assessment': assessment
    }

if __name__ == "__main__":
    detailed_phonetic_analysis()
