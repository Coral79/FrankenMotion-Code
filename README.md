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
- **[Coming Soon]** Pre-trained model checkpoints and training code.

## 📦 Release Progress
- [x] **Frankenstein Dataset**: The first motion dataset with asynchronous, part-level text annotations.
- [ ] **Model Weights**: Pre-trained checkpoints for FrankenMotion.
- [ ] **Core Code**: Training and inference pipeline.
- [ ] **Evaluation**: Evaluation code and metrics.

---

## 📄 Abstract
Human motion generation from text prompts has made remarkable progress in recent years. However, existing methods primarily rely on either sequence-level or action-level descriptions due to the absence of fine-grained, part-level motion annotations. This limits their controllability over individual body parts.

In this work, we construct a high-quality motion dataset with atomic, temporally-aware part-level text annotations, leveraging the reasoning capabilities of large language models (LLMs). Unlike prior datasets, our dataset captures asynchronous and semantically distinct part movements at fine temporal resolution. Based on this dataset, we introduce **FrankenMotion**, a diffusion-based part-aware motion generation framework where each body part is guided by its own temporally-structured textual prompt. Experiments demonstrate that FrankenMotion outperforms previous baseline models and can compose motions unseen during training.

---

## 🧟 Frankenstein Dataset

<details>
<summary><b>Click to expand</b></summary>

The Frankenstein Dataset contains **10,978** motion sequences from [AMASS](https://amass.is.tue.mpg.de/) with fine-grained, per-body-part text annotations. Each sequence includes independent text descriptions for 7 body parts (head, spine, left/right arm, left/right leg, trajectory) plus global captions, with precise temporal boundaries.

Sequence-level labels are sourced and enriched from [BABEL](https://babel.is.tue.mpg.de/) and [HumanML3D](https://github.com/EricGuo5513/HumanML3D) annotations, extended with fine-grained part-level descriptions generated with LLM assistance.

**Download from HuggingFace:** [Coral79/frankenstein-dataset](https://huggingface.co/datasets/Coral79/frankenstein-dataset)

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

### 3. Preprocess AMASS data

First, obtain the SMPL-H body model by following the [README from TEMOS](https://github.com/Mathux/TEMOS?tab=readme-ov-file#4-optional-smpl-body-model) to get the `deps` folder, and place it in the root directory.

Then run the preprocessing pipeline:

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

Pre-computed CLIP ViT-B/32 embeddings are included in the dataset. To recompute them yourself:

```bash
pip install clip
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

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Coral79/FrankenMotion-Code&type=Date)](https://star-history.com/#Coral79/FrankenMotion-Code&Date)
