# PHNet‑CMAPSS · Mission Control

A local web dashboard for the NASA CMAPSS predictive‑maintenance research
framework (adaptive attention fusion of 4 autoencoders, sensor Graph
Attention Network, HI‑guided cross‑attention, physics‑informed multi‑task
loss, self‑supervised pretraining, EVT/POT adaptive thresholding,
adaptive gating ensemble, and Integrated‑Gradients explainability).

Press **RUN PIPELINE** in the browser and the full 11‑stage pipeline runs
in a background thread on your machine, streaming live logs and
rendering metrics, gauges, charts, an ablation table, and a sensor
importance ranking the moment it finishes.

## What's inside

```
app.py                 Flask backend — starts/monitors the run, serves the UI
pipeline/
  config.py            Two presets: "quick" (fast demo) and "full" (paper-scale)
  data.py               CMAPSS loading; falls back to synthetic data if
                         the real dataset can't be downloaded
  models.py              Every architectural piece: GAT, cross-attention,
                          4 autoencoders (MC-Dropout), attention fusion,
                          dynamic health index, physics-informed loss,
                          self-supervised pretraining, EVT threshold,
                          Integrated Gradients
  runner.py              Orchestrates all 11 stages end-to-end and
                          renders matplotlib figures to base64 (no files
                          written to disk)
templates/index.html      Dashboard markup
static/css/style.css      Instrument-panel visual design
static/js/app.js           Polling, live checklist, SVG gauges, charts
requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`torch`, `xgboost`, and `kagglehub` are the heavier installs — everything
else is lightweight.

### Optional: real NASA CMAPSS data

The dashboard tries to download the real dataset via `kagglehub`
(`behrad3d/nasa-cmaps`) on every run. To use the real data:

1. Create a free Kaggle account and an API token
   (Kaggle → Account → **Create New API Token**, downloads `kaggle.json`).
2. Place it at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` /
   `KAGGLE_KEY` environment variables).

If no credentials/internet are available, the pipeline **automatically
falls back** to a physically-motivated synthetic FD001-like dataset with
the identical schema, so the button always produces a real result — the
dashboard clearly labels which data source was used ("Run summary" →
"Data source").

## Run it

```bash
python3 app.py
```

Then open **http://localhost:5050** in your browser.

- **Preset → Quick demo**: small model dimensions, 2–3 epochs, finishes
  in well under a minute on a laptop CPU. This is the default and is
  meant to always complete fast so you can see the whole pipeline work.
- **Preset → Full research**: the original paper hyperparameters
  (`SEQ_LEN=30`, `D_MODEL=64`, 30 fine-tuning epochs, 20 MC-Dropout
  samples, etc.). This is compute-heavy — use a GPU box and expect a
  materially longer run.

Click **RUN PIPELINE**. The left-hand checklist lights up stage by
stage, the **Live Log** tab streams every training/eval line as it
happens, and **Overview / Charts / Ablation / Sensor Ranking** populate
automatically the moment the run finishes.

## Notes

- Everything runs **locally, on your machine** — no data leaves your
  computer, and no external services are called except the optional
  Kaggle download and Google Fonts (for the UI typefaces).
- The server keeps at most one run in flight; starting a second run
  while one is active returns `409` until the first finishes.
- Figures are generated in-memory as base64 PNGs and streamed straight
  into the page — nothing is written to disk unless you add that
  yourself.
