#!/usr/bin/env python3
"""
Direct comparison of manu.wav files using the Ainu audio analysis system
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('src')
from audio_utils import load_audio, trim_silence, extract_features, zscore, dtw_distance, cosine_similarity_matrix

def compare_manu_files():
    """Compare the two manu.wav files directly"""
    
    # File paths
    recording_path = "/home/nk21137/OneDrive/5years/graduation_research/ainu/data/recordings/yabuki/manu.wav"
    sample_path = "/home/nk21137/OneDrive/5years/graduation_research/ainu/data/samples/vacabulary/manu.wav"
    
    print("🎵 Ainu Audio Analysis: Comparing 'manu' pronunciations")
    print("=" * 60)
    print(f"Recording: {recording_path}")
    print(f"Sample:    {sample_path}")
    print()
    
    # Audio parameters
    TARGET_SR = 16000
    N_MFCC = 13
    HOP_LENGTH = 160
    N_FFT = 400
    TRIM_DB = 30.0
    
    try:
        # Load recording
        print("📂 Loading recording...")
        rec_audio = load_audio(recording_path, target_sr=TARGET_SR, mono=True)
        rec_duration = rec_audio.y.shape[0] / rec_audio.sr
        print(f"   Duration: {rec_duration:.2f}s @ {rec_audio.sr} Hz")
        
        # Load sample
        print("📂 Loading sample...")
        samp_audio = load_audio(sample_path, target_sr=TARGET_SR, mono=True)
        samp_duration = samp_audio.y.shape[0] / samp_audio.sr
        print(f"   Duration: {samp_duration:.2f}s @ {samp_audio.sr} Hz")
        print()
        
        # Trim silence
        print("✂️  Trimming silence...")
        rec_trimmed = trim_silence(rec_audio, top_db=TRIM_DB)
        samp_trimmed = trim_silence(samp_audio, top_db=TRIM_DB)
        
        rec_trim_dur = rec_trimmed.y.shape[0] / rec_trimmed.sr
        samp_trim_dur = samp_trimmed.y.shape[0] / samp_trimmed.sr
        print(f"   Recording trimmed: {rec_trim_dur:.2f}s")
        print(f"   Sample trimmed:    {samp_trim_dur:.2f}s")
        print()
        
        # Extract features
        print("🔬 Extracting features...")
        rec_features = extract_features(rec_trimmed, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, 
                                      n_fft=N_FFT, add_deltas=True, add_prosody=True)
        samp_features = extract_features(samp_trimmed, n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
                                       n_fft=N_FFT, add_deltas=True, add_prosody=True)
        
        print(f"   Recording features: {rec_features.shape} (frames × features)")
        print(f"   Sample features:    {samp_features.shape} (frames × features)")
        print()
        
        # Normalize features
        print("📊 Normalizing features...")
        rec_norm = zscore(rec_features)
        samp_norm = zscore(samp_features)
        print()
        
        # Calculate DTW similarity
        print("🧮 Calculating DTW similarity...")
        try:
            dtw_result = dtw_distance(rec_norm, samp_norm, metric='cosine')
            if isinstance(dtw_result, tuple) and len(dtw_result) == 2:
                dtw_dist, dtw_path = dtw_result
            else:
                dtw_dist = dtw_result
                dtw_path = None
        except:
            # Fallback DTW calculation
            from dtw import dtw
            import librosa
            cost_matrix = 1.0 - cosine_similarity_matrix(rec_norm, samp_norm)
            alignment = dtw(cost_matrix)
            dtw_dist = alignment.distance
            dtw_path = list(zip(alignment.index1, alignment.index2))
        
        # Normalize DTW score by sequence lengths
        total_frames = rec_norm.shape[0] + samp_norm.shape[0]
        dtw_score = dtw_dist / max(total_frames, 1)
        
        print(f"   DTW distance: {dtw_dist:.6f}")
        print(f"   DTW score (normalized): {dtw_score:.8f}")
        print()
        
        # Calculate frame-wise cosine similarity
        print("📈 Calculating cosine similarity matrix...")
        cos_sim_matrix = cosine_similarity_matrix(rec_norm, samp_norm)
        mean_cos_sim = np.mean(cos_sim_matrix)
        max_cos_sim = np.max(cos_sim_matrix)
        
        print(f"   Mean cosine similarity: {mean_cos_sim:.4f}")
        print(f"   Max cosine similarity:  {max_cos_sim:.4f}")
        print()
        
        # Analysis results
        print("🎯 ANALYSIS RESULTS")
        print("=" * 30)
        print(f"Word:                 'manu' (bear)")
        print(f"Recording duration:   {rec_trim_dur:.2f}s")
        print(f"Sample duration:      {samp_trim_dur:.2f}s")
        print(f"DTW alignment score:  {dtw_score:.8f} (lower = more similar)")
        print(f"Cosine similarity:    {mean_cos_sim:.4f} (higher = more similar)")
        
        # Interpretation
        print()
        print("📝 INTERPRETATION")
        print("=" * 20)
        if dtw_score < 0.001:
            similarity_level = "EXCELLENT"
        elif dtw_score < 0.01:
            similarity_level = "VERY GOOD"
        elif dtw_score < 0.1:
            similarity_level = "GOOD"
        elif dtw_score < 1.0:
            similarity_level = "MODERATE"
        else:
            similarity_level = "LOW"
            
        print(f"Pronunciation similarity: {similarity_level}")
        
        if mean_cos_sim > 0.8:
            acoustic_match = "EXCELLENT acoustic match"
        elif mean_cos_sim > 0.6:
            acoustic_match = "GOOD acoustic match"
        elif mean_cos_sim > 0.4:
            acoustic_match = "MODERATE acoustic match"
        else:
            acoustic_match = "LOW acoustic match"
            
        print(f"Acoustic characteristics: {acoustic_match}")
        print()
        
        # Create visualization
        print("📊 Creating visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Ainu 'manu' Pronunciation Comparison", fontsize=14)
        
        # Waveforms
        axes[0,0].plot(np.linspace(0, rec_trim_dur, len(rec_trimmed.y)), rec_trimmed.y, 'b-', alpha=0.7, label='Recording')
        axes[0,0].set_title('Recording Waveform')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(np.linspace(0, samp_trim_dur, len(samp_trimmed.y)), samp_trimmed.y, 'r-', alpha=0.7, label='Sample')
        axes[0,1].set_title('Sample Waveform')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Amplitude')
        axes[0,1].grid(True, alpha=0.3)
        
        # Cosine similarity matrix
        im = axes[1,0].imshow(cos_sim_matrix, origin='lower', aspect='auto', cmap='viridis')
        axes[1,0].set_title('Cosine Similarity Matrix')
        axes[1,0].set_xlabel('Sample frames')
        axes[1,0].set_ylabel('Recording frames')
        plt.colorbar(im, ax=axes[1,0])
        
        # DTW path
        cost_matrix = 1.0 - cos_sim_matrix
        axes[1,1].imshow(cost_matrix, origin='lower', aspect='auto', cmap='magma')
        if dtw_path is not None:
            try:
                if len(dtw_path) > 0 and len(dtw_path[0]) == 2:
                    path_x, path_y = zip(*dtw_path)
                    axes[1,1].plot(path_y, path_x, 'cyan', linewidth=2, alpha=0.8)
            except:
                pass  # Skip path plotting if there's an issue
        axes[1,1].set_title('DTW Cost Matrix & Alignment')
        axes[1,1].set_xlabel('Sample frames')
        axes[1,1].set_ylabel('Recording frames')
        
        plt.tight_layout()
        
        # Save results
        output_dir = Path('outputs/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_dir / 'manu_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved: {plot_path}")
        
        # Save comparison results
        results = {
            'recording_file': recording_path,
            'sample_file': sample_path,
            'recording_duration': rec_trim_dur,
            'sample_duration': samp_trim_dur,
            'dtw_distance': dtw_dist,
            'dtw_score': dtw_score,
            'mean_cosine_similarity': mean_cos_sim,
            'max_cosine_similarity': max_cos_sim,
            'similarity_level': similarity_level,
            'acoustic_match': acoustic_match
        }
        
        results_df = pd.DataFrame([results])
        csv_path = output_dir / 'manu_comparison_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"   Results saved: {csv_path}")
        
        plt.show()
        print()
        print("✅ Analysis completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

if __name__ == "__main__":
    compare_manu_files()
