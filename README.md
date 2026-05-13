# FrankenMotion: Part-level Human Motion Generation and Composition

<p align="center">
  <a href="https://coral79.github.io/frankenmotion/"><b>[🌐 Project Page]</b></a>
  <a href="https://arxiv.org/abs/2601.10909"><b>[📄 Paper]</b></a>
  <a href="https://huggingface.co/datasets/Coral79/frankenstein-dataset"><b>[🧟 Dataset]</b></a>
</p>

This is the official repository for **FrankenMotion**.

---

## 🚀 News
- **[Jan 2026]** Our paper has been submitted to arXiv!
- **[Feb 2026]** FrankenMotion is accepted at CVPR 2026, see you in Denver!
- **[Apr 2026]** The **Frankenstein Dataset** is now available on [HuggingFace](https://huggingface.co/datasets/Coral79/frankenstein-dataset)!
- **[May 2026]** Pre-trained model checkpoint, training & inference code, and evaluation pipeline are now released!

## 📦 Release Progress
- [x] **Frankenstein Dataset**: The first motion dataset with asynchronous, part-level text annotations.
- [x] **FrankenAgent**: LLM-based annotation pipeline used to build the dataset.
- [x] **Model Weights**: Pre-trained checkpoints for FrankenMotion.
- [x] **Core Code**: Training and inference pipeline.
- [x] **Evaluation**: Evaluation code and metrics.

---

## 📄 Abstract
Human motion generation from text prompts has made remarkable progress in recent years. However, existing methods primarily rely on either sequence-level or action-level descriptions due to the absence of fine-grained, part-level motion annotations. This limits their controllability over individual body parts.

In this work, we construct a high-quality motion dataset with atomic, temporally-aware part-level text annotations, leveraging the reasoning capabilities of large language models (LLMs). Unlike prior datasets, our dataset captures asynchronous and semantically distinct part movements at fine temporal resolution. Based on this dataset, we introduce **FrankenMotion**, a diffusion-based part-aware motion generation framework where each body part is guided by its own temporally-structured textual prompt. Experiments demonstrate that FrankenMotion outperforms previous baseline models and can compose motions unseen during training.

---

## ⚙️ Installation

Tested with **Python 3.9**, **PyTorch 2.5.1**, **CUDA 11.8** (CPU also supported).

```bash
# 1. Clone the repo
git clone https://github.com/Coral79/FrankenMotion-Code.git
cd FrankenMotion-Code

# 2. Create the conda env (Python, PyTorch+CUDA, ffmpeg, all pip deps, OpenAI CLIP)
conda env create -f environment.yml
conda activate frankenmotion
```

---

## 🧟 Frankenstein Dataset

<details>
<summary><b>Click to expand</b></summary>

The Frankenstein Dataset contains **16,097** motion sequences from [AMASS](https://amass.is.tue.mpg.de/) with fine-grained, per-body-part text annotations. Each sequence includes independent text descriptions for 7 body parts (head, spine, left/right arm, left/right leg, trajectory) plus global captions, with precise temporal boundaries.

Sequence-level labels are sourced and enriched from [BABEL](https://babel.is.tue.mpg.de/) and [HumanML3D](https://github.com/EricGuo5513/HumanML3D) annotations, extended with fine-grained part-level descriptions generated with LLM assistance.

**Download from HuggingFace:** [Coral79/frankenstein-dataset](https://huggingface.co/datasets/Coral79/frankenstein-dataset)

> **Note:** If you downloaded the dataset before **April 26, 2026**, please re-download — the current version corrects an over-aggressive sample filter.

### 1. Download dataset annotations

```bash
# Option A: git clone
git clone https://huggingface.co/datasets/Coral79/frankenstein-dataset datasets/annotations/frankenstein-dataset

# Option B: huggingface_hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Coral79/frankenstein-dataset', repo_type='dataset', local_dir='datasets/annotations/frankenstein-dataset')
"
```

### 2. Download AMASS motion data

The motion sequences are from [AMASS](https://amass.is.tue.mpg.de/) and cannot be redistributed. Please download all **SMPL-H G format** motions from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) and place them in `datasets/motions/AMASS/`.

### 3. Obtain the SMPL-H body model

SMPL-H is licence-gated and cannot be redistributed. Please follow the [README from TEMOS](https://github.com/Mathux/TEMOS?tab=readme-ov-file#4-optional-smpl-body-model) to register, download, and place the `deps/` folder at the repo root. After this step you should have `deps/smplh/SMPLH_NEUTRAL.npz` (and male/female variants).

### 4. Preprocess AMASS data

```bash
python prepare/prepare_amass.py --amass_dir datasets/motions/AMASS --smplh_dir deps/smplh
```

This runs the full pipeline (fix FPS → mirror → extract joints → compute SMPL-RiFKE features) and produces a 205-dimensional representation at 20 FPS.

<details>
<summary>Click for details on each preprocessing step</summary>

- **Fix FPS**: Interpolates SMPL pose parameters to 20.0 FPS and removes hand parameters. Output: `AMASS_20.0_fps_nh/`
- **Mirror**: Mirrors motions for data augmentation (as in HumanML3D). Output: `AMASS_20.0_fps_nh/M/`
- **Extract joints**: Computes 24 SMPL joint positions using the SMPL-H layer. Output: `AMASS_20.0_fps_nh_smpljoints_neutral_nobetas/`
- **SMPL-RiFKE**: Combines joints + 6D pose into a 205-dim unified representation. Output: `AMASS_20.0_fps_nh_smplrifke/`

</details>

### 5. (Optional) Recompute CLIP embeddings

Pre-computed CLIP ViT-B/32 embeddings are included in the dataset. To recompute them yourself, make sure OpenAI CLIP is installed (it is included in `environment.yml`), then run:

```bash
python prepare/compute_clip_embeddings.py \
    --dataset_dir datasets/annotations/frankenstein-dataset
```

### Expected directory structure

After setup:

```
datasets/
  annotations/
    frankenstein-dataset/
      annotations/
        annotations.json
        splits/
          train.txt  val.txt  test.txt
      text_embeddings/
        clip/
          clip.npy  clip_index.json  clip_slice.npy
  motions/
    AMASS/
    AMASS_20.0_fps_nh/
    AMASS_20.0_fps_nh_smpljoints_neutral_nobetas/
    AMASS_20.0_fps_nh_smplrifke/
deps/
  smplh/
```

</details>

---

## 🤖 FrankenAgent

<details>
<summary><b>Click to expand</b></summary>

The Frankenstein Dataset is built using **FrankenAgent**, an LLM-based annotation pipeline that converts existing motion annotations (BABEL + HumanML3D + KIT-ML, unified via the [AMASS-Annotation-Unifier](https://github.com/Mathux/AMASS-Annotation-Unifier)) into per-body-part text descriptions with temporal boundaries.

The released annotations were produced with **DeepSeek-R1-0521**. The pipeline is dataset-agnostic: a single Python script + two prompts.

See [`FrankenAgent/README.md`](FrankenAgent/README.md) for the input format, prompt selection, and how to annotate a new motion dataset.

</details>

---

## 📥 Pre-trained Models

Two HuggingFace repos host the pre-trained model checkpoints used by inference and evaluation. Download both up front so the rest of the README's commands work without further setup.

```bash
# 1. FrankenMotion diffusion checkpoint (~170 MB) — used by inference, training resume, and evaluation
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Coral79/frankenmotion', local_dir='pretrained/frankenmotion')
"

# 2. Per-body-part retrieval encoders (~335 MB) — used by the evaluation protocols
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Coral79/frankenmotion-eval-model', local_dir='pretrained/eval_model')
"
```

---

## 🎬 Inference

<details>
<summary><b>Click to expand</b></summary>

> **Prerequisites.** Complete §🧟 Frankenstein Dataset **step 1** (dataset annotations + CLIP text-embedding cache) and **step 3** (SMPL-H body model at `deps/smplh/`), and §📥 Pre-trained Models (FrankenMotion checkpoint), before running the commands below.

### Option 1 (default): sample from the dataset val split

```bash
python generate_part.py num_sample=10
```

This generates and renders the first 10 val keyids under `pretrained/frankenmotion/generations/t2m/val_samples_10_frankenmotion_val_to_motion/`. Both the model's prediction and the ground-truth motion are rendered for side-by-side comparison. Increase `num_sample` to write more samples, or pass `fast=true` to write only the joint trajectories without rendering videos.

Per-sample output files:
- `val_samples_<N>_val_<i>.npy` — predicted 24-joint positions per frame
- `val_samples_<N>_val_<i>_smpl.npz` — predicted SMPL parameters
- `val_samples_<N>_val_<i>_smpl.mp4` — predicted-motion visualisation
- `val_samples_<N>_val_<i>_smpl_gt.mp4` — ground-truth visualisation (same prompt, real motion)
- `val_samples_<N>_val_<i>_input_text.txt` — input ground-truth per-body-part captions used to condition the model

### Option 2: generate from a custom prompt

To generate a motion from your own text prompt instead of sampling the dataset, prepare a JSON describing per-body-part text + temporal segments and pass `input_type=user_input`:

```bash
python generate_part.py \
    input_type=user_input \
    user_input_file=test_inputs/sit_look_up.json \
    num_sample=1
```

The bundled `test_inputs/sit_look_up.json` shows the expected schema:

```json
{
    "duration": 4.0,
    "sequence_caption": "unknown",
    "body_parts": {
        "left_leg":   [{"start": 0.0, "end": 4.0, "text": "sit"}],
        "right_leg":  [{"start": 0.0, "end": 4.0, "text": "sit"}],
        "left_arm":   [{"start": 0.0, "end": 4.0, "text": "unknown"}],
        "right_arm":  [{"start": 0.0, "end": 4.0, "text": "unknown"}],
        "head":       [{"start": 0.0, "end": 4.0, "text": "look up"}],
        "spine":      [{"start": 0.0, "end": 4.0, "text": "unknown"}],
        "trajectory": [{"start": 0.0, "end": 4.0, "text": "unknown"}]
    }
}
```

Segments within a body part may be asynchronous and need not span the full duration. **The input can be sparse** — body parts you don't want to constrain can either be omitted from `body_parts` entirely or marked as `"unknown"` (as the arms, spine, trajectory, and even the `sequence_caption` are above); the model fills in plausible motion for those parts.

Output lands under `pretrained/frankenmotion/generations/t2m/<json_basename>_frankenmotion_user_input_to_motion/`.

</details>

---

## 🏋️ Training

> **Prerequisites — datasets and SMPL-H.** Training requires the full preprocessed AMASS motion features at `datasets/motions/AMASS_20.0_fps_nh_smplrifke/` plus the dataset annotations at `datasets/annotations/frankenstein-dataset/`. Complete §🧟 Frankenstein Dataset steps 1–4 first (download + preprocess). SMPL-H is also needed at `deps/smplh/`.

<details>
<summary><b>Recommended setup (matches the pre-trained checkpoint)</b></summary>

| Knob | Default value | Notes |
|---|---|---|
| Hardware | 1 × NVIDIA H100 80 GB | V100 also works, ~30 % slower; smaller GPUs need `dataloader.batch_size` reduced |
| Batch size | 64 | `dataloader.batch_size=64` (override for smaller GPUs) |
| Workers | 8 | `dataloader.num_workers=8` |
| Epochs | 6000 | `trainer.max_epochs=6000` |
| Optimizer | AdamW, lr 1e-4 | set in `configs/mdm_smpl_part.yaml` |
| Loss | WeightedMSE | `model.loss_type=WeightedMSELoss` |
| Text encoder | CLIP ViT-B/32, 8-part body order, `rand_mask=true`, `mask_ratio=0.5` | set in `configs/text_encoder/clip_part.yaml` |
| Motion features | smplrifke, 613-dim with global trajectory + action + head | set in `configs/motion_loader/smplrifke.yaml` |
| Checkpointing | every epoch (`save_top_k=1` + `save_last`) + every 500 epochs (`save_top_k=-1`) | set in `configs/trainer.yaml` |
| Wall-clock | ~48 h on 1 × H100 for the full 6000 epochs | first-run dataset preload adds ~25 min |

**These values are all defaults** — `python train_part.py dataset=frankenstein-dataset` reproduces the pre-trained checkpoint setup verbatim. Override only when departing from the reference configuration.

</details>

### Command

```bash
python train_part.py dataset=frankenstein-dataset
```

This trains the model to convergence using the recommended setup above.

### Outputs

`outputs_amass/mdm-smpl_clip_part_smplrifke_frankenstein-dataset/` contains:
- `logs/checkpoints/{last,latest-epoch=*}.ckpt` — Lightning checkpoints
- `motion_stats/{mean,std}.pt`, `text_stats/{mean,std}.pt` — normalisation stats (computed once on the train split, reused on resume)
- `config.json` — the resolved Hydra config for the run

To resume from an interrupted run, point `run_dir=<previous_output_dir>` and pass `ckpt=last`.

<details>
<summary><b>Best-epoch tracking (optional)</b></summary>

`configs/trainer.yaml` includes an `EvalInTraining` callback (off by default) that periodically runs the full eval pipeline mid-training and appends one row per fired epoch to `{run_dir}/eval_in_training/summary.csv`. Useful for picking the best epoch in retrospect, since model quality typically peaks mid-training rather than at the final epoch.

```bash
# Enable in-training eval every 200 epochs (val split, 1 turn, cuda)
python train_part.py dataset=frankenstein-dataset trainer.callbacks.4.enabled=true

# Slower, higher-fidelity numbers (every 100 ep, 3 turns, cpu)
python train_part.py dataset=frankenstein-dataset \
    trainer.callbacks.4.enabled=true \
    trainer.callbacks.4.every_n_epochs=100 \
    trainer.callbacks.4.num_turns=3 \
    trainer.callbacks.4.device=cpu
```

</details>

---

## 📊 Evaluation

<details>
<summary><b>Click to expand</b></summary>

> **Prerequisites.** Complete §🧟 Frankenstein Dataset (all five steps) and §📥 Pre-trained Models (both downloads) before running the commands below.

> Generation (`eval_part.py`, `generate_part.py`) runs on GPU by default. Only metric computation (`src.eval.run` in step (b)) is pinned to **cpu** — the cpu pass is the reference for the reported numbers.

### Run the evaluation

For full paper reproduction use `num_runs=20` / `--max_runs 20`; use `3` for a quicker run.

```bash
# (a) Generate motions for the test split, repeated 20 times
python eval_part.py \
    run_dir=pretrained/frankenmotion \
    ckpt=frankenmotion \
    num_runs=20 \
    video_samples_per_run=2

# (b) Compute the paper table on CPU — auto-runs both retrieval protocols and saves a single paper_table.csv
EXP_DIR=$(ls -dt pretrained/frankenmotion/generations/t2m/test_frankenmotion_*/ | head -1)
python -m src.eval.paper_table "${EXP_DIR%/}" \
    --max_runs 20 \
    --dataset_name frankenstein-dataset \
    --device cpu
```

Results are written to `${EXP_DIR}/paper_table.csv` and should match paper Table 1 within run-to-run noise.

</details>

---

## 👀 You Might Also Like

Also check out our arXiv 2026 paper 🎯 **[ActionPlan](https://coral79.github.io/ActionPlan/)** — future-aware streaming motion synthesis via frame-level action planning, enabling real-time generation **5.25× faster** with better quality.

---

## ✍️ Citation
If you find our work or code useful for your research, please consider citing:

```bibtex
@inproceedings{li2026frankenmotion,
  title={{FrankenMotion}: Part-level Human Motion Generation and Composition},
  author={Li, Chuqiao and Xie, Xianghui and Cao, Yong and Geiger, Andreas and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 📜 License

This project is released under a non-commercial research license. See [LICENSE](LICENSE) for details.

The motion data from AMASS is subject to the [AMASS license](https://amass.is.tue.mpg.de/license.html) and must be obtained separately.

### Acknowledgements

The motion preprocessing pipeline and SMPL-RiFKE representation are adapted from:
- [STMC](https://github.com/nv-tlabs/stmc) (Petrovich et al., CVPRW 2024)
- [TMR](https://github.com/Mathux/TMR) (Petrovich et al., ICCV 2023)
- [UniMotion](https://github.com/Coral79/Unimotion) (Li et al., 3DV 2025)

Dataset annotations build on labels from:
- [BABEL](https://babel.is.tue.mpg.de/) (Punnakkal et al., CVPR 2021)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) (Guo et al., CVPR 2022)
- [KIT Motion-Language Dataset](https://motion-annotation.humanoids.kit.edu/dataset/) (Plappert et al., 2016)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Coral79/FrankenMotion-Code&type=Date)](https://star-history.com/#Coral79/FrankenMotion-Code&Date)
