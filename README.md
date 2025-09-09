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
