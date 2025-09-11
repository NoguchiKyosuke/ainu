# Ainu Speech Analysis

This workspace contains a Jupyter-based system to analyze recorded audio and compare it with a sample Ainu audio set.

## Project Structure
```
ainu/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── venv/                       # Virtual environment (auto-created)
├── data/
│   ├── recordings/             # Your audio recordings
│   └── samples/                # Reference Ainu audio samples
├── notebooks/
│   └── 01_ainu_audio_compare.ipynb  # Main analysis notebook
├── src/
│   ├── audio_utils.py          # Audio processing utilities
│   └── plots.py                # Visualization utilities
└── outputs/                    # Analysis results and cache (auto-created)
    ├── cache/                  # Cached feature files
    └── results/                # Analysis outputs
```

## Setup Status ✅

### Environment Setup (Completed)
- ✅ **Python 3.12.3** detected and compatible
- ✅ **Virtual environment** created at `./venv/`
- ✅ **All dependencies installed** from requirements.txt:
  - Core scientific stack (numpy, scipy, pandas, matplotlib, seaborn)
  - Audio processing (librosa, soundfile)
  - Machine learning (scikit-learn, DTW)
  - Jupyter environment with widgets

### Jupyter Server (Running)
- ✅ **Jupyter server active** at `http://127.0.0.1:8888`
- ✅ **Authentication token** configured for secure access
- ✅ **VS Code Simple Browser** opened for easy access

### Quick Start
1. **Activate environment** (if not already active):
   ```bash
   source venv/bin/activate
   ```

2. **Start Jupyter** (if not already running):
   ```bash
   jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
   ```

3. **Open main notebook**: Navigate to `notebooks/01_ainu_audio_compare.ipynb`

### Adding Audio Data
- Place your **recorded audio** in `data/recordings/`
- Place **reference Ainu samples** in `data/samples/` (organized by label in subfolders)
- Supported formats: WAV, FLAC, MP3, OGG, M4A (16 kHz recommended)

## Analysis Pipeline

### Features
- **Audio preprocessing**: Loading, resampling, silence trimming with VAD
- **Feature extraction**: MFCC + deltas + spectral features + simple prosody
- **Similarity metrics**: 
  - DTW (Dynamic Time Warping) with cosine distance
  - Optional neural embeddings (Wav2Vec2)
- **Ranking system**: Combined scoring with configurable weights
- **Visualization**: Waveforms, spectrograms, similarity matrices, DTW paths

### Main Notebook Workflow
1. **Load and index** sample audio dataset from folder structure
2. **Process recorded audio** with silence trimming and normalization
3. **Extract features** using MFCC and spectral analysis
4. **Calculate similarity** using DTW alignment and optional embeddings
5. **Rank matches** and visualize results with interactive plots

## Notebook `01_ainu_audio_compare.ipynb` Usage Guide

The main notebook now consolidates all functionality (data indexing, similarity scoring, direct file comparison, advanced phonetic analysis, interactive UI). Run cells in order—earlier cells define constants and helper functions used later.

### Section Overview (Headings in the Notebook)
| Section | Purpose |
|---------|---------|
| Setup & Imports | Environment checks, library imports, configuration constants |
| Data Indexing / Listing | Scans `data/samples/` and `data/recordings/` into DataFrames |
| Feature Extraction | Loads audio, trims silence, computes MFCC + deltas + prosody features |
| Similarity & DTW | Computes cosine similarity matrices and DTW alignment scores |
| Batch Comparison | Ranks all samples against a recording with cached features |
| Direct File Comparison | One-off pronunciation comparison between two specific `.wav` files |
| Visualization Suite | Waveforms, spectrograms, MFCC trends, similarity matrices, DTW path |
| Advanced Phonetic Analysis | Spectral centroid, rolloff, bandwidth, ZCR, pitch (YIN), energy, tempo |
| Interactive Analysis Widget | Dropdown-driven UI + audio playback + report generation |

### Quick Start Inside the Notebook
1. Open the notebook and run all setup/import cells (top of the file).
2. Place your recording in `data/recordings/` (e.g. `data/recordings/manu.wav`).
3. Ensure at least one reference sample exists in `data/samples/` (nested folders OK).
4. Use either:
   - Batch comparison section to rank all samples, or
   - Direct file comparison for a specific pair.
5. (Optional) Run advanced phonetic analysis for deeper acoustic feature similarity.
6. (Optional) Use the interactive widget (requires `ipywidgets`).

### Direct File Comparison (Pronunciation Similarity)
Run in a code cell (adjust paths):
```python
results = compare_specific_files(
  'data/recordings/manu.wav',
  'data/samples/Some_Story/manu.wav',
  word='manu'
)
visualize_comparison(results)
```
Outputs:
- DTW distance + normalized score (lower = better)
- Mean / max cosine similarity (higher = better)
- Visual panels (waveforms, spectrograms, MFCC trend, similarity matrix, DTW path)

### Advanced Phonetic Analysis
After obtaining `results`:
```python
phonetic = detailed_phonetic_comparison(results)
create_phonetic_report(results, phonetic, save_path='outputs/results/manu_phonetic_report.md')
```
This computes and reports:
- Spectral Centroid / Rolloff / Bandwidth
- Zero Crossing Rate & RMS Energy
- Fundamental Frequency (F0) via YIN
- Similarity scores + weighted overall phonetic similarity

### Interactive Analyzer (Widget)
If `ipywidgets` is enabled:
```python
analyzer_widget  # Created automatically in the widget section
```
Features:
- Dropdown selection for recording & sample files
- Audio playback (browser-supported formats)
- One-click visualization & DTW + phonetic report generation

If widgets are unavailable, fallback instructions print selectable file paths and manual function calls.

### Output Artifacts
| Path | Description |
|------|-------------|
| `outputs/results/*.png` | Visualization images (e.g. `manu_comparison.png`) |
| `outputs/results/*_results.csv` | Tabular similarity summaries |
| `outputs/results/*_phonetic_report.md` | Markdown phonetic reports |
| `outputs/cache/*.npz` | Cached feature arrays to speed repeat analyses |

### Interpreting Scores
| Metric | Interpretation |
|--------|----------------|
| DTW Score | Normalized alignment cost; <0.01 typically very similar |
| Mean Cosine Similarity | Frame-wise acoustic similarity (0–1) |
| Phonetic Similarity % | Weighted spectral + pitch + energy feature agreement |

General guidance:
- High cosine + low DTW = strong temporal & acoustic match.
- High phonetic similarity with moderate DTW may indicate timing differences (e.g., pacing) but similar articulation.

### Adding New Recordings
1. Save a `.wav` file (mono preferred) into `data/recordings/`.
2. Re-run the data indexing cell (or just the direct comparison cell with the new filename).
3. Use the widget or functions above for analysis.

### Extending the Notebook
Ideas:
- Add embedding-based similarity (e.g., Wav2Vec2) and blend with DTW.
- Export JSON summary per comparison for downstream apps.
- Integrate a Streamlit or Gradio front-end using functions in `src/`.

---
If you add new sections to the notebook, update this guide so users can locate functionality quickly.

## Source Code Explanation (Notebook & Core Utilities)

This section explains the main functions defined in the notebook (`01_ainu_audio_compare.ipynb`) and supporting module `src/audio_utils.py`.

### Core Data Structures
- Raw audio container (from `load_audio`):
  - `y`: 1D NumPy array (float32 waveform, mono)
  - `sr`: Sample rate (int)
  - Additional metadata may include trimmed variants
- Feature matrix (from `extract_features`): shape `(T, F)` where:
  - `T` = number of frames
  - `F` = MFCCs + deltas + prosody/spectral features (variable, typically 13–40+)
- Similarity matrix: shape `(T_rec, T_ref)` cosine similarities in [-1, 1]
- DTW path: list of `(i, j)` index pairs aligning recording → sample frames

### Utility Module (`src/audio_utils.py`)
| Function | Purpose | Key Params | Returns |
|----------|---------|-----------|---------|
| `load_audio(path, target_sr, mono)` | Load & resample audio | path, target_sr | object with `y`, `sr` |
| `trim_silence(audio, top_db)` | Remove leading/trailing silence | dB threshold | trimmed audio object |
| `extract_features(audio, n_mfcc, hop_length, n_fft, add_deltas, add_prosody)` | Build feature matrix | config constants | `(T, F)` array |
| `zscore(features)` | Per-feature standardization | features | normalized array |
| `cosine_similarity_matrix(A, B)` | Framewise similarity | 2 matrices | `(T_A, T_B)` matrix |
| `dtw_distance(A, B, metric)` | Temporal alignment cost | feature matrices | distance (and optional path) |

### Notebook-Defined Functions
| Function | Role |
|----------|------|
| `compare_specific_files(recording_path, sample_path, word)` | Orchestrates end‑to‑end comparison: load → trim → feature extract → normalize → DTW + cosine metrics + interpretation |
| `visualize_comparison(result_dict, save_path)` | Multi-panel plot: waveforms, spectrograms, MFCC mean trace, cosine similarity heatmap, DTW path, summary panel |
| `analyze_phonetic_features(audio, sr, ...)` | Extracts extended phonetic descriptors (centroid, rolloff, bandwidth, ZCR, RMS, F0 via YIN, mel spec, chroma, tempo, onsets) |
| `detailed_phonetic_comparison(result_dict)` | Compares aggregated statistics (means) between recording and reference; computes per-feature similarity scores + weighted overall score |
| `create_phonetic_report(results, phonetic_analysis, save_path)` | Builds Markdown report table + interpretation block and writes to disk |
| `create_interactive_analyzer()` | Builds ipywidgets UI (dropdowns + buttons + audio players) enabling ad‑hoc comparisons without manual code |

### Result Dictionary (from `compare_specific_files`)
```python
{
  'recording_file': str,
  'sample_file': str,
  'word': str,
  'recording_duration': float,  # seconds (trimmed)
  'sample_duration': float,
  'dtw_distance': float,
  'dtw_score': float,           # normalized distance
  'mean_cosine_similarity': float,
  'max_cosine_similarity': float,
  'similarity_level': str,      # heuristic label
  'acoustic_match': str,        # heuristic label
  'features': { 'recording': np.ndarray, 'sample': np.ndarray },
  'audio': { 'recording': audio_obj, 'sample': audio_obj },
  'cos_sim_matrix': np.ndarray,
  'dtw_path': list[(i, j)] or None
}
```

### Phonetic Analysis Output (from `detailed_phonetic_comparison`)
```python
{
  'recording_features': {...},        # raw per-frame & summary stats
  'sample_features': {...},
  'feature_comparisons': {
     feature_name: {
        'recording_mean': float,
        'sample_mean': float,
        'absolute_difference': float,
        'relative_difference_percent': float,
        'similarity_score': float  # 0–100
     }, ...
  },
  'overall_phonetic_similarity': float,  # weighted %
  'similarity_weights': { feature: weight }
}
```

### Processing Flow (Pseudocode)
```python
def pipeline(recording, sample):
    rec = load_audio(recording, target_sr)
    ref = load_audio(sample, target_sr)
    rec = trim_silence(rec, top_db)
    ref = trim_silence(ref, top_db)
    R = zscore(extract_features(rec, ...))
    S = zscore(extract_features(ref, ...))
    dtw_dist, path = dtw_distance(R, S, metric)
    sim_mat = cosine_similarity_matrix(R, S)
    return aggregate_metrics(R, S, dtw_dist, path, sim_mat)
```

### Error Handling & Fallbacks
- DTW path absence: code checks tuple vs scalar return and degrades gracefully.
- DTW failure: falls back to manual distance via cost matrix + `dtw` alignment.
- Missing files: early existence checks with user-facing messages.
- Pitch extraction: ignores zero/unvoiced frames when computing F0 stats.

### Performance Considerations
- Feature caching recommended for batch operations (`outputs/cache/`).
- Keep hop length moderate (e.g., 256–512) to balance time resolution vs speed.
- DTW cost grows O(T_rec * T_ref); pre-trimming silence reduces compute.

### Extensibility Hooks
- Add embedding extraction (e.g., Wav2Vec2) and merge with DTW via weighted fusion.
- Insert language-specific phoneme segmentation before feature aggregation.
- Replace heuristic similarity labels with learned thresholds from annotated data.

### Quality Tips
- Ensure consistent sample rate (rely on `load_audio` resampling).
- Always normalize per-utterance (`zscore`) before similarity to mitigate loudness variance.
- Inspect MFCC mean trace divergence to spot timing vs articulation mismatches.

This explanation should help contributors modify or extend analysis logic confidently.

## Development Notes
- **Audio format**: WAV (16 kHz recommended). Other formats supported via soundfile.
- **Feature pipeline**: MFCCs + spectral features → per-utterance z-score normalization
- **Similarity**: DTW with cosine distance for temporal alignment
- **Caching**: Features saved to `outputs/cache/` for faster re-analysis
- **Cultural sensitivity**: Handle Ainu language data with appropriate respect for cultural heritage

## Troubleshooting

### Common Issues
- **soundfile errors on Linux**: Install system dependency
  ```bash
  sudo apt-get install -y libsndfile1
  ```
- **Memory issues with large datasets**: Use feature caching and batch processing
- **Jupyter widgets not working**: Enable extension
  ```bash
  jupyter nbextension enable --py widgetsnbextension
  ```

### Optional Features
- **Neural embeddings**: Uncomment torch/transformers lines in requirements.txt
- **GPU acceleration**: Install CUDA-compatible PyTorch for faster embedding extraction
