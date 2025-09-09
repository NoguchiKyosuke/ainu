# Ainu Speech Analysis

This workspace contains a Jupyter-based system to analyze recorded audio and compare it with a sample Ainu audio set.

## Folders
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `data/` — Put your audio here
  - `data/samples/` — Reference (Ainu) audio set
  - `data/recordings/` — Your recorded audio
- `src/` — Reusable Python modules

## Setup
1. Create and activate a Python 3.10+ environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. (Recommended) Enable Jupyter widgets:
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   ```

## First steps
Open the notebook `notebooks/01_ainu_audio_compare.ipynb` and follow the instructions.

## Notes
- Audio expected format: WAV (16 kHz recommended). Other formats may work via soundfile.
- The pipeline extracts MFCCs + spectral features, then compares via cosine similarity and DTW.
- You can swap in neural embeddings later if desired.
