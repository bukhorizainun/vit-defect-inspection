# ViT Defect Inspection

A **Vision Transformer (ViT) built from scratch in PyTorch** for surface-defect
classification, with **attention-rollout visualisation** that shows where the
model looks when it flags a crack. The target use case is automated visual
inspection, where a model has to tell an intact surface from a defective one and,
ideally, point at the defect.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bukhorizainun/vit-defect-inspection/blob/main/notebooks/vit_crack_colab.ipynb)

## Why a Vision Transformer

A ViT splits an image into fixed-size patches, embeds each patch, prepends a
class token, and passes the sequence through Transformer encoder blocks. Because
attention is global from the first layer, the model can relate a thin crack on
one side of an image to context on the other side. The self-attention weights
also give a natural way to visualise *what the model attends to* — useful when a
human inspector needs to trust the decision.

This repository implements the ViT itself (patch embedding, class token,
positional embeddings, multi-head self-attention, Transformer blocks) rather than
calling a black-box model, so the mechanics are visible and easy to modify. A
pretrained baseline via [`timm`](https://github.com/huggingface/pytorch-image-models)
is also available for comparison.

## What it does

- Classifies surface images as **cracked / intact** (any two-class or multi-class
  `ImageFolder` dataset works).
- Visualises decisions with **attention rollout**, which highlights the patches
  that drive the prediction — typically the defect region.
- Trains from scratch, or fine-tunes a pretrained ViT with a single flag.

## Repository layout

```
src/
  vit.py         from-scratch Vision Transformer
  attention.py   attention-rollout + overlay visualisation
  data.py        ImageFolder loaders and transforms
  train.py       training / evaluation loop and metrics
tests/
  smoke_test.py  offline check (random data, no download)
notebooks/
  vit_crack_colab.ipynb   end-to-end training on Colab
```

## Quickstart

**Colab (recommended):** open the notebook with the badge above. It clones this
repo, downloads the public Surface Crack dataset, trains the ViT, and renders
attention maps.

**Local:**

```bash
pip install -r requirements.txt

# offline sanity check (no dataset needed)
python -m tests.smoke_test

# train on an ImageFolder dataset with Positive/ and Negative/ subfolders
python -m src.train --data-dir /path/to/surface_crack --epochs 15

# or fine-tune a pretrained ViT
python -m src.train --data-dir /path/to/surface_crack --pretrained --epochs 5
```

## Results

Trained on the [Surface Crack Detection dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
(≈40k images, 227×227, `Positive`/`Negative`). Numbers are filled in from the
Colab run.

| Model | Params | Val accuracy |
|-------|-------:|-------------:|
| ViT (from scratch, depth 6) | ~2.9M | _to be added_ |
| ViT-tiny (pretrained, timm) | ~5.7M | _to be added_ |

Attention-rollout examples are saved to `assets/` by the notebook.

## Dataset

Surface Crack Detection by Ç. F. Özgenel, mirrored on Kaggle
(`arunrk7/surface-crack-detection`). The dataset is downloaded at runtime and is
not stored in this repository.

## License

MIT — see [LICENSE](LICENSE).
