# Copilot Instructions for this Repository

Purpose: Build a Jupyter-based system to analyze recorded speech and compare it to a sample Ainu audio set, as groundwork for a future training app.

Project stack
- Python 3.10+
- Jupyter Notebooks (VS Code)
- Libraries: librosa, soundfile, numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, tqdm, ipywidgets
- Optional embeddings: PyTorch, torchaudio, transformers
- Repo structure:
  - `notebooks/01_ainu_audio_compare.ipynb` main workflow
  - `src/audio_utils.py` core audio utilities (loading, trimming, features, cosine sim, DTW)
  - `src/plots.py` plotting helpers
  - `data/samples/` reference Ainu audio (organized by label in subfolders)
  - `data/recordings/` user recordings
  - `outputs/` results and cache

Conventions and best practices
- Audio
  - Target sample rate: 16 kHz, mono.
  - Use `src/audio_utils.load_audio` (handles resampling/mono) and `trim_silence` for VAD-like trimming.
  - Features: `extract_features` (MFCC + deltas + simple prosody), then per-utterance z-score with `zscore`.
- Similarity
  - Framewise cosine similarity via `cosine_similarity_matrix`.
  - Alignment: `dtw_distance` (uses librosa.sequence.dtw on a cosine-distance cost matrix). Lower distance = better.
- Outputs
  - Save cached features to `outputs/cache/` and results (CSV/plots) to `outputs/results/`.
- Notebooks
  - Prefer calling functions in `src/` instead of duplicating logic in the notebook.
  - Keep cells deterministic and guard optional dependencies (torch/transformers) with try/except.

What to generate
- Utilities and notebook cells that:
  - Respect the directory layout above.
  - Use existing helpers in `src/` where possible.
  - Are concise, with clear variable names and minimal side effects.
- If adding new features, consider placing reusable code in `src/` and importing it in notebooks.

What to avoid
- Duplicating logic already in `src/audio_utils.py`.
- Hard-coding file paths outside of `data/` and `outputs/`.
- Long-running code without progress bars or clear messages.
- Committing large binary audio files.

Common tasks Copilot should assist with
- Indexing `data/samples/` and building a labeled DataFrame from folder names.
- Comparing a chosen recording against all samples with DTW scoring and producing a ranked table.
- Visualizing waveforms, mel-spectrograms, similarity matrices, and DTW paths.
- Caching features to NPZ and reusing them.
- Optional: extracting model embeddings and combining scores (DTW + cosine).

Quality and style
- Keep answers short and impersonal.
- When asked for your name, respond with: "GitHub Copilot".
- Prefer readable, commented code. Small, testable functions.
- Use bash-compatible commands for Linux when suggesting shell steps.

Setup reminders
- Install: `pip install -r requirements.txt`
- If soundfile errors on Linux: `sudo apt-get install -y libsndfile1`
- Enable widgets (optional): `jupyter nbextension enable --py widgetsnbextension`

Safety and data
- Do not include or generate copyrighted data/content.
- Be mindful of cultural and language data sensitivity; respect dataset licenses.

Extension ideas
- Wrap comparison in a small Streamlit/Gradio app using `src/audio_utils`.
- Add unit tests for feature extraction and DTW scoring.
