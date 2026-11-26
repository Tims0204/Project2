# Self-Implemented ViT on CIFAR-10

A Vision Transformer implemented from scratch in PyTorch and evaluated on CIFAR-10.
The project studies the effect of model capacity (hidden dimension, depth) and data
augmentation, and uses Shapley values to analyse which image patches the models rely on.

Layer normalisation, multi-head attention and the cross-entropy loss are implemented by hand.
The optimiser is `torch.optim.AdamW`.  <!-- adjust if utils/optimizer.py contains your own AdamW -->

## Structure

```
model/
  embeddings.py            # patch embedding, class token, positional embedding
  layernorm.py             # hand-written layer normalisation
  transformer_encoder.py   # attention, MLP, Transformer blocks
  classifierHead.py        # classification head
  vit.py, model.py         # full ViT model (FullModel)
dataset/
  data.py                  # loading CIFAR-10, normalisation, augmentation
  helper.py                # data helpers
utils/
  loss.py                  # hand-written cross-entropy loss
  optimizer.py             # optimiser code
analysis/
  shapely_values.py        # patch-wise Shapley values + seed stability
  plots/                   # heatmaps and saved values
train.py                   # training script
run_all.sh                 # runs all experiments
summary.py                 # aggregates runs into summary.csv
summary.csv                # results table (mean / std over seeds)
viz.ipynb                  # plots and visualisations
runs/                      # checkpoints: <cfg>_s<seed>.pt, e.g. hd96_nh6_d2_s42.pt
logs/                      # training logs
latex/                     # report source (main.tex, Report.tex, references.bib)
requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The dataset (`uoft-cs/cifar10`) is loaded via Hugging Face `datasets`.
A CUDA GPU is recommended.

## Training

Run all experiments (24 configurations x 3 seeds):

```bash
bash run_all.sh
```

Single run (see `python train.py --help` for all arguments):

```bash
python train.py --optimizer AdamW --lr 5e-4 --weight_decay 0.05 <further args>
```

Shared hyperparameters: patch size 4, 6 heads, MLP factor 4, batch size 512,
100 epochs, 5 warm-up epochs followed by cosine decay. The configurations vary
hidden dimension D in {96, 192, 384, 768}, depth L in {2, 4, 6}, augmentation on/off,
and seeds 42, 43, 44.

## Results summary

```bash
python summary.py    # writes summary.csv
```

## Shapley values

Patch-wise Shapley values (8x8 patches, Monte Carlo estimate with M samples per patch).
Run from the project root so that `model` and `dataset` can be imported:

```bash
PYTHONPATH=. python analysis/shapely_values.py
```

The configuration is set in the `__main__` block. Alternatively, in Python:

```python
from analysis.shapely_values import run_config

# single checkpoint
run_config(ckpts="runs/hd96_nh6_d2_s42.pt", n_images=20, M=200)

# all seeds of one configuration (also computes Spearman correlation between seeds)
run_config("hd96_nh6_d2", seeds=(42, 43, 44), n_images=20, M=200)
```

Heatmaps and raw values (`<cfg>_phi_seeds.pt`) are written to `analysis/plots/`.

## Report

```bash
cd latex
pdflatex main && biber main && pdflatex main && pdflatex main
```

## Main results (100 epochs, mean over 3 seeds)

| Setting | Test accuracy |
|---|---|
| With augmentation, D=768, L=4 | 88.68 % |
| Without augmentation, D=768, L=6 | 71.01 % |
| D=768, L=6, 300 epochs (single run) | 91.54 % |

See the report for the full table and the Shapley analysis.