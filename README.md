# FrankenMotion: Part-level Human Motion Generation and Composition

<p align="center">
  <a href="https://coral79.github.io/frankenmotion/"><b>[🌐 Project Page]</b></a>
  <a href="https://arxiv.org/abs/2601.XXXXX"><b>[📄 Paper]</b></a>
</p>

This is the official repository for **FrankenMotion**. 

---

## 🚀 News
- **[Jan 2026]** Our paper has been submitted to arXiv!
- **[Feb 2026]** FrankenMotion is accepted at CVPR 2026, see you in Denver!
- **[Coming Soon]** We will release the full code, pre-trained models, and the **Frankenstein Dataset**. 
- **⭐ Star us** to get notified when the code is released!


## 📦 Content to be Released
- [ ] **Core Code**: Training and inference pipeline.
- [ ] **Model Weights**: Pre-trained checkpoints for FrankenMotion.
- [ ] **Frankenstein Dataset**: The first motion dataset with asynchronous, part-level text annotations.

## 📄 Abstract
Human motion generation from text prompts has made remarkable progress in recent years. However, existing methods primarily rely on either sequence-level or action-level descriptions due to the absence of fine-grained, part-level motion annotations. This limits their controllability over individual body parts.

In this work, we construct a high-quality motion dataset with atomic, temporally-aware part-level text annotations, leveraging the reasoning capabilities of large language models (LLMs). Unlike prior datasets, our dataset captures asynchronous and semantically distinct part movements at fine temporal resolution. Based on this dataset, we introduce **FrankenMotion**, a diffusion-based part-aware motion generation framework where each body part is guided by its own temporally-structured textual prompt. Experiments demonstrate that FrankenMotion outperforms previous baseline models and can compose motions unseen during training.

## ✍️ Citation
If you find our work or code useful for your research, please consider citing:

```bibtex
@article{li2026frankenmotion,
  title={{FrankenMotion}: Part-level Human Motion Generation and Composition},
  author={Li, Chuqiao and Xie, Xianghui and Cao, Yong and Geiger, Andreas and Pons-Moll, Gerard},
  journal={arXiv preprint arXiv:2601.10909},
  year={2026}
}
