# ASL GISLR — PyTorch Baseline (Transformer) + Gemini Sentence Builder

This is a PyTorch implementation for isolated ASL word recognition on Kaggle's **GISLR** dataset,
including an inference demo that turns a stream of recognized words into a **simple sentence** via **Google Generative AI (Gemini)**.

## Highlights
- Landmark-based pipeline (MediaPipe schema: 543 points).
- Preprocess step that: (1) selects lips + dominant hand + small pose subset, (2) flips coordinates to a **left-dominant** canonical form, (3) filters frames with no hand, (4) downsamples/pads to **64** frames using edge padding + uniform average pooling.
- Transformer encoder (2 blocks, 8 heads, 384 dim) with GELU MLP.
- Label smoothing (0.25), AdamW with cosine schedule and optional warmup, weight decay tied to LR (wd = wd_ratio * lr).
- Balanced per-class sampling during training.
- Webcam demo using MediaPipe Holistic and Gemini to re-order recognized words into a simple sentence.

## Quickstart

```bash
# 0) Create env (Python >=3.10 recommended)
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# or: .venv\Scripts\activate (Windows)

# 1) Install deps
pip install -r requirements.txt
(Be sure to install the torch version that matches your NVIDIA Driver.)

# 3) Train
python scripts/train.py 

# 5) Webcam inference (+ Gemini sentence)
# Requires GOOGLE_API_KEY in a .env file
python inference/webcam_demo.py --checkpoint ./checkpoints/best.pt
```

See **scripts/config.py** for all hyperparameters.