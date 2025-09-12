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

### Notebook Cell 9: Sample Dataset Indexing

Cell 9 in the notebook defines the logic that enumerates and labels all reference (sample) audio files. It performs these steps:

1. Defines `infer_label_from_path(p)` which assigns a class/label using the parent directory name of each file (folder-based labeling convention: `data/samples/<label>/<file>.wav`).
2. Calls `list_audio_files(SAMPLES_DIR)` to recursively gather supported audio files (WAV/FLAC/etc.).
3. Warns the user if no files are found so the workflow can be halted early.
4. Builds a pandas DataFrame `samples_df` with columns:
  - `path`: absolute or project-relative path string
  - `label`: inferred category (story / speaker / token grouping)
  - `relpath`: path relative to `data/samples/` (useful for display & exporting)
5. Prints the total count of indexed files and displays the head (first 10 rows) for verification.

Purpose in pipeline:
- Establishes the searchable reference corpus used in later DTW / cosine similarity ranking.
- Supplies labels for grouping, filtering, or stratified evaluation.

Typical Output Example:
```
Indexed 2816 sample files.
        path              label                relpath
0  data/samples/StoryA/...    StoryA    StoryA/file1.wav
1  data/samples/StoryA/...    StoryA    StoryA/file2.wav
...
```

If you reorganize `data/samples/`, re-run Cell 9 to rebuild `samples_df` before performing comparisons.

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

## Montreal Forced Aligner (MFA) Integration (Prototype)

This repository now includes a scaffold to build forced alignments for the Ainu folklore corpus at: https://ainu.ninjal.ac.jp/folklore/corpus/jp/

### Why Alignments?
Forced alignment yields time-stamped word/phone boundaries (TextGrid files) enabling:
- Segment-level pronunciation feedback (timing & substitution detection)
- Vowel length / gemination analysis
- Precise pitch & formant extraction per phone segment

### Directory Layout
```
mfa/
  resources/
    ainu_language.yaml   # MFA language config (phoneset + parameters, draft)
    ainu_base.dict       # Seed dictionary (manual overrides)
    ainu.dict            # Generated merged dictionary (output of builder)
  corpus_prep/
    prepare_corpus.py    # Builds MFA corpus/ with .lab transcripts
  dictionary_builder.py  # Generates ainu.dict from corpus + base entries
  run_mfa.sh             # Helper script for prepare/dict/align/train
```

### Install MFA
```
pip install montreal-forced-aligner
mfa version
```
If system dependencies are required (e.g. Kaldi tools), consult MFA docs: https://montreal-forced-aligner.readthedocs.io/

### Step 1: Prepare Corpus
Organize or point to your audio root (e.g. `data/samples/folklore`). Then:
```
bash mfa/run_mfa.sh prepare data/samples/folklore mfa/corpus
```
Outputs a symlinked corpus tree with `.lab` transcripts (placeholder text = `UNKNOWN` if no mapping supplied). Provide a CSV mapping for real text:
```
relpath,transcription
Story1/utt001.wav,manu kor an ...
Story1/utt002.wav, ...
```
Run with mapping:
```
python mfa/corpus_prep/prepare_corpus.py --audio-root data/samples/folklore --out-dir mfa/corpus --mapping transcripts.csv
```

### Step 2: Build Dictionary
```
bash mfa/run_mfa.sh dict mfa/corpus mfa/resources/ainu.dict
```
Merges manual `ainu_base.dict` with heuristic G2P over observed corpus words.

### Step 3: Align
```
bash mfa/run_mfa.sh align mfa/corpus mfa/resources/ainu.dict mfa/resources/ainu_language.yaml mfa/aligned
```
Results:
- TextGrid files in `mfa/aligned/` with word & (if supported) phone tiers

### (Optional) Step 4: Train Acoustic Model
```
bash mfa/run_mfa.sh train mfa/corpus mfa/resources/ainu.dict mfa/resources/ainu_language.yaml mfa/model
```
Produces a reusable MFA acoustic model for faster future alignment.

### Improving Quality
- Replace placeholder transcripts (`UNKNOWN`) with accurate text.
- Expand `ainu_base.dict` with authoritative phonemic transcriptions.
- Refine phoneset in `ainu_language.yaml` (long vowels, gemination, affricates, glottal stop, dialectal variants).
- Add disambiguation for homographs if needed.

### Using Alignments in Notebooks
After alignment, parse TextGrid boundaries to:
- Slice features per phone and compute phone-level DTW or distance metrics.
- Provide targeted feedback: which segment deviated most.
- Normalize durations: compare expected vs produced segment length.

Example (future work):
```python
import tgt
grid = tgt.read_textgrid('mfa/aligned/Story1/utt001.TextGrid')
phones = [ti for ti in grid.get_tier_by_name('phones')]  # iterate intervals
```

### Next Steps (Planned)
1. Integrate TextGrid parser utility in `src/`.
2. Map aligned phones back into feedback JSON for segment-level coaching.
3. Add composite score combining DTW + segment accuracy + duration variance.

Contribution welcome for phoneme inventory refinement and verified dictionary entries.

## Folklore Corpus Downloader (Prototype)

A helper script is included to assist in locally mirroring metadata and (where permissible) audio references from the Ainu folklore corpus website for research preparation. Use responsibly.

Path: `mfa/download_folklore_corpus.py`

### IMPORTANT LEGAL / ETHICAL NOTICE
- Check the site's Terms of Use and robots.txt before large-scale crawling.
- Download only what you need; apply rate limiting (default delay is 1.5s between requests).
- Do not redistribute downloaded audio or texts without explicit permission.
- Cite the original source in any research outputs.

### Install Additional Dependencies
`requirements.txt` now includes:
```
requests
beautifulsoup4
```
Install (if not already):
```
pip install -r requirements.txt
```

### Basic Usage
```
python mfa/download_folklore_corpus.py --out data/raw_folklore --delay 2.0
```
Creates structure:
```
data/raw_folklore/
  index_links.csv
  pages/                 # raw HTML of story pages
  audio/                 # per-story audio (if direct links discoverable)
  metadata/              # per-story JSON metadata
  stories_metadata.csv   # aggregated metadata
```

### Options
| Flag | Description |
|------|-------------|
| `--limit N` | Stop after first N stories (debugging) |
| `--delay S` | Seconds between HTTP requests (throttle) |
| `--overwrite` | Redownload & replace existing audio/pages |
| `--index-only` | Skip audio downloads; just build link & metadata index |

### Resuming
Re-run with same output directory; existing audio is skipped unless `--overwrite` is set.

### After Download
1. Inspect `stories_metadata.csv` to confirm coverage.
2. Manually curate transcriptions or build a mapping CSV for `prepare_corpus.py`.
3. Run MFA corpus prep + dictionary build + alignment.

### Limitations
- Audio links may be embedded or require scripted interactions not handled yet.
- Transcription extraction is heuristic (grabs large text blocks); refine with site-specific parsing if needed.
- Japanese text retained; no automatic romanization performed.

### Future Improvements
- Add multi-page traversal if pagination exists.
- More robust audio URL detection (JS player reversal).
- Configurable metadata field extraction.

Contributions refining respectful crawling patterns or adding canonical citation templates are welcome.
