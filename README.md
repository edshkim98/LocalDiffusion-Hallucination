<div align="center">

# Tackling Structural Hallucination in Image Translation with Local Diffusion

**European Conference on Computer Vision (ECCV) 2024 — Oral Presentation**

[![Paper](https://img.shields.io/badge/arXiv-2404.05980-b31b1b.svg)](https://arxiv.org/abs/2404.05980)
[![Conference](https://img.shields.io/badge/ECCV-2024%20Oral-4b44ce.svg)](https://eccv.ecva.net/)
[![Python](https://img.shields.io/badge/Python-3.9-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-ee4c2c.svg)](https://pytorch.org/)

Seunghoi Kim · Chen Jin · Tom Diethe · Matteo Figini · Henry F. J. Tregidgo · Asher Mullokandov · Philip Teare · Daniel C. Alexander

</div>

---

A **training-free** diffusion framework that reduces *structural hallucination* when translating out-of-distribution (OOD) images — such as unseen tumors in medical scans — with pre-trained diffusion models. It plugs into an existing model without any retraining and cuts misdiagnosis by **40%** (medical) and **25%** (natural images).

<div align="center">
<img src="imgs/intro2.png" width="85%" alt="Hallucination example"/>
</div>

## Table of Contents

- [Background](#background)
- [Why It Works](#why-it-works)
- [Method](#method)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Background

Recent developments in diffusion models have advanced conditioned image generation, yet they struggle to reconstruct out-of-distribution (OOD) images, such as unseen tumors in medical images. This causes **"image hallucination"** and risks misdiagnosis. We hypothesize that **hallucinations are caused by local OOD regions in the conditional image** — and that by partitioning the OOD region from the in-distribution (IND) region and generating them separately, hallucinations can be alleviated.

## Why It Works

A conditional diffusion model trained only on in-distribution data has never seen the OOD structure. When it is given a whole conditional image that contains such a region, its learned prior has no valid way to explain those pixels — so it maps them onto the nearest structure it *does* know, producing a plausible-but-wrong reconstruction. **That is the hallucination.**

Branching removes the conflict rather than trying to correct it after the fact:

- **The OOD branch** is conditioned *only* on the OOD region, so the model is never asked to reconcile anomalous pixels with a global in-distribution context — it simply reconstructs that patch.
- **The IND branch** is conditioned on the in-distribution region, where the prior is valid and reconstruction stays faithful.

Neither branch is ever placed in the regime that produces hallucination, and **fusion** stitches the two clean predictions back into a single seamless image. In short: instead of fixing hallucination after it happens, local diffusion **never puts the model in the failure regime in the first place** — which is also why it needs no retraining.

## Method

<div align="center">
<img src="imgs/intro1.png" width="85%" alt="Method overview"/>
</div>

The pipeline has three stages, applied entirely at inference time on top of a frozen, pre-trained diffusion model:

| Stage | What happens | Code |
| :---- | :----------- | :--- |
| **1. OOD detection** | A [PatchCore](https://arxiv.org/abs/2106.08265) anomaly detector (or, optionally, a pre-trained segmentation network for MRI) scores the conditional image and produces a soft OOD-probability mask. | `compute_ood_mask` — `test.py` |
| **2. Branching** | The reverse diffusion runs as **two parallel branches** — one conditioned only on the OOD region, one on the IND region — so the model never has to explain the unfamiliar region with in-distribution structures. | `branching_out`, `model_predictions` — `ddpm.py` |
| **3. Fusion** | Once `t ≤ start_timestep`, the two branch predictions are merged with the OOD mask and the remaining reverse steps run jointly, yielding one coherent image. An optional PatchCore **classifier** can adaptively decide whether to accept the fused result or keep branching. | `fusion` — `ddpm.py` |

## Results

The method reduces misdiagnosis by **40%** and **25%** on real-world medical and natural-image datasets respectively, and integrates with existing pre-trained diffusion models without any retraining.

<div align="center">
<img src="imgs/supple1.png" width="85%" alt="Qualitative results (medical)"/>
<img src="imgs/supple2.png" width="85%" alt="Qualitative results (natural images)"/>
</div>

## Installation

Python ≥ 3.9 with a CUDA-capable GPU is recommended.

```bash
pip install -r requirements.txt
```

> **Note** — `anomalib` (pinned to `0.3.7` for its legacy module layout) is only required at **test** time for OOD detection. Training has no dependency on it.

## Quick Start

The MNIST demo — train on digit **8**, test on OOD digit **3** — runs out of the box, since the raw MNIST files ship with the repo:

```bash
# 1. Train the diffusion model on in-distribution data (digit 8)
python train.py --config configs/mnist_train.yaml

# 2. Build the PatchCore memory bank used by the OOD detector
python anomaly_model_train.py --mode mnist

# 3. Test with local diffusion on OOD data (digit 3)
python test.py --config configs/mnist_test.yaml
```

## Usage

### 1. Train the diffusion model on in-distribution data

```bash
python train.py --config configs/mnist_train.yaml     # MNIST 2× super-resolution
python train.py --config configs/brats_train.yaml     # BRATS T1 → FLAIR translation
```

Checkpoints (`model-best<step>.pt`), the config snapshot, loss curves, and sample outputs are written to `./results/<ProjectName>/`.

### 2. Pre-compute the PatchCore memory bank (OOD detector)

The OOD detector needs a memory bank of in-distribution features:

```bash
python anomaly_model_train.py --mode mnist
python anomaly_model_train.py --mode mri   --mri_files '/path/to/BRATS_png/normal/*flair.png'
python anomaly_model_train.py --mode mvtec --mvtec_files './mvtec/transistor/*/good/*.png' --obj transistor
```

Point `memory_bank_path` in your test config at the resulting `memory_bank_*.npy`.
For MRI you may instead use a pre-trained tumor-segmentation network: run `python train_seg.py`, then set `ood_detector.seg: True` and `ood_detector.seg_model` in the test config.

### 3. Test with local diffusion

```bash
python test.py --config configs/mnist_test.yaml
python test.py --config configs/brats_test.yaml
python test.py --config configs/mvtec_test.yaml
```

Ensure `ProjectName` and `train_phase` in the test config point at the trained checkpoint (`./results/<ProjectName>/model-best<train_phase>.pt`). The script saves the following to `./results/<ProjectName>/test_outputs/`:

| File | Contents |
| :--- | :------- |
| `hr_all.npy`   | ground-truth images |
| `lr_all.npy`   | conditional inputs |
| `pred_all.npy` | model predictions |
| `ad_masks.npy` | detected OOD masks |

## Configuration

Every entry point is driven by a YAML file in [`configs/`](configs/) (each key is commented inline). The flags that control the method:

| Key | Meaning |
| :-- | :------ |
| `ood` | test on OOD samples (`True`) or IND samples (`False`) |
| `ood_AD` | detect the OOD region automatically with PatchCore |
| `branch_out` | run the separate IND / OOD reverse-diffusion branches |
| `start_intermediate` / `start_timestep` | fuse the two branches once `t ≤ start_timestep` |
| `mask_x` | mask the OOD-branch prediction outside the OOD region |
| `classifier` | adaptive fusion: accept/reject the fused prediction with a PatchCore classifier |
| `ddim_timestep` | set `< timestep` for DDIM sampling, or `False` for full DDPM |

> **Baseline** — setting `ood_AD: False`, `branch_out: False`, and `start_intermediate: False` recovers the vanilla conditional diffusion model.

> **Calibration** — the OOD-mask thresholds in `compute_ood_mask` (`test.py`) are the per-dataset values calibrated for the paper's experiments. A new dataset requires calibrating an entry threshold on its PatchCore anomaly scores.

## Repository Structure

```
├── train.py                 # train the conditional diffusion model (IND data only)
├── test.py                  # test with local diffusion (OOD detection + branching + fusion)
├── anomaly_model_train.py   # pre-compute the PatchCore memory bank used by the OOD detector
├── configs/                 # YAML configs for every entry point (commented inline)
├── ddpm.py                  # U-Net, Gaussian diffusion (branching/fusion sampling), Trainer
├── data.py                  # datasets: MNIST, BRATS (MedDataset*), MVTec
├── models.py                # PatchCore OOD detector + adaptive-fusion classifier
├── unet_model.py            # conditional-image encoder (ResUnet) and segmentation U-Net
├── utils.py                 # shared helpers (config loading, model builder, seeding)
├── train_seg.py             # optional: train the MRI tumor-segmentation OOD detector
├── train_mnist_cls.py       # optional: MNIST classifier for the misdiagnosis evaluation
└── MNIST/                   # raw MNIST (bundled, so the demo runs out of the box)
```

Developed with Python 3.9.5 and `torch==1.12.1+cu113`. See [`requirements.txt`](requirements.txt) for the full list.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{kim2024tackling,
  title     = {Tackling Structural Hallucination in Image Translation with Local Diffusion},
  author    = {Kim, Seunghoi and Jin, Chen and Diethe, Tom and Figini, Matteo and Tregidgo, Henry F. J. and Mullokandov, Asher and Teare, Philip and Alexander, Daniel C.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
