"""Convenience loader that returns a `tmr_forward(motions_or_texts)` closure.

Used by `src.eval.run` to load a TMR checkpoint directory (produced by
tmr_plus training) and get a single callable that encodes both texts and
motions into the shared TMR latent space.
"""

from __future__ import annotations

import os

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.config import read_config

from .collate import collate_x_dict
from .ckpt_loader import load_model_from_cfg, rewrite_targets


def _resolve_tmr_path(run_dir: str, raw_path: str) -> str:
    """Resolve a path that may contain the unresolved Hydra interpolation
    ``${hydra:runtime.choices.data}`` (or the motion-loader variant). Falls
    back to the two-levels-up-from-run_dir layout used by tmr_plus.
    """
    if "${hydra:" in raw_path:
        # Strip Hydra runtime vars. For TMR checkpoint folders named like
        # `tmr_<dataset>_<motion-loader>_<part>`, the dataset and motion-loader
        # are encoded in the folder name: dataset = 2nd field, motion-loader
        # = 3rd field (these are the slug-forms of the choices).
        folder = os.path.basename(os.path.normpath(run_dir))
        parts = folder.split("_")
        # tmr_<dataset>_<motion-loader>_<anything>
        dataset_slug = parts[1] if len(parts) >= 2 else ""
        motion_loader_slug = parts[2] if len(parts) >= 3 else ""
        raw_path = raw_path.replace("${hydra:runtime.choices.data/motion_loader}", motion_loader_slug)
        raw_path = raw_path.replace("${hydra:runtime.choices.data}", dataset_slug)

    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    if os.path.exists(raw_path):
        return raw_path
    # Release layout: all encoders share a sibling ``stats/`` directory; text
    # embeddings (if present) live at ``<root>/datasets/...``.
    run_parent = os.path.dirname(os.path.normpath(run_dir))
    tail = raw_path.split(os.sep)[0]  # "stats" or "datasets"
    sibling = os.path.join(run_parent, tail)
    if os.path.isdir(sibling):
        return sibling
    # Legacy tmr_plus layout: stats/ and datasets/ sit two levels above run_dir.
    legacy = os.path.join(os.path.dirname(run_parent), raw_path)
    if os.path.exists(legacy):
        return legacy
    return raw_path  # downstream will fail with a useful message


def load_eval_model(device: str = "cpu", run_dir: str | None = None):
    if run_dir is None:
        run_dir = "pretrained/tmr/tmr_humanml3d_augmented"
    ckpt_name = "last"

    cfg = read_config(run_dir)
    cfg.run_dir = run_dir
    rewrite_targets(cfg)

    # The stored configs reference a Hydra runtime variable that doesn't resolve
    # here; patch both normalizer.base_dir and text_to_token_emb.path to
    # concrete paths we can find from disk.
    norm_cfg = cfg.data.motion_loader.normalizer
    raw_base = OmegaConf.to_container(norm_cfg, resolve=False)["base_dir"]
    norm_cfg.base_dir = _resolve_tmr_path(run_dir, raw_base)

    text_cfg = cfg.data.text_to_token_emb
    raw_text_path = OmegaConf.to_container(text_cfg, resolve=False)["path"]
    text_cfg.path = _resolve_tmr_path(run_dir, raw_text_path)

    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    normalizer = instantiate(cfg.data.motion_loader.normalizer)
    text_model = instantiate(cfg.data.text_to_token_emb, preload=False, device=device)

    def easy_forward(motions_or_texts):
        if isinstance(motions_or_texts[0], str):
            x_dict = collate_x_dict(text_model(motions_or_texts))
        else:
            motions = [
                normalizer(torch.from_numpy(m).to(torch.float)).to(device)
                for m in motions_or_texts
            ]
            x_dict = collate_x_dict(
                [{"x": m, "length": len(m)} for m in motions]
            )

        with torch.inference_mode():
            latents = model.encode(x_dict, sample_mean=True).cpu()
        return latents

    return easy_forward
