"""Load TMR checkpoints into the TMR module layout bundled in this repo.

TMR checkpoints were trained with the stand-alone ``tmr_plus`` codebase, whose
Hydra configs bake in ``_target_`` strings under ``src.model.*`` /
``src.data.*``. We rewrite those pointers at load time so the same checkpoints
can be consumed from ``src.eval.eval_model.*`` here without editing files on disk.
"""

from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import read_config

logger = logging.getLogger(__name__)


# Maps original tmr_plus targets to the targets in this repo.
_TARGET_REMAP = {
    "src.model.TMR": "src.eval.eval_model.model.TMR",
    "src.model.TEMOS": "src.eval.eval_model.temos.TEMOS",
    "src.model.ACTORStyleEncoder": "src.eval.eval_model.actor.ACTORStyleEncoder",
    "src.model.ACTORStyleDecoder": "src.eval.eval_model.actor.ACTORStyleDecoder",
    "src.model.PositionalEncoding": "src.eval.eval_model.actor.PositionalEncoding",
    "src.model.TextToEmb": "src.eval.eval_model.text_encoder.TextToEmb",
    "src.model.losses.InfoNCE_with_filtering": "src.eval.eval_model.losses.InfoNCE_with_filtering",
    "src.model.losses.HN_InfoNCE_with_filtering": "src.eval.eval_model.losses.HN_InfoNCE_with_filtering",
    "src.model.losses.KLLoss": "src.eval.eval_model.losses.KLLoss",
    "src.data.motion.Normalizer": "src.eval.eval_model.data.motion.Normalizer",
    "src.data.motion.AMASSMotionLoader": "src.eval.eval_model.data.motion.AMASSMotionLoader",
    "src.data.text.TokenEmbeddings": "src.eval.eval_model.data.text.TokenEmbeddings",
    "src.data.text.SentenceEmbeddings": "src.eval.eval_model.data.text.SentenceEmbeddings",
}


def rewrite_targets(cfg: DictConfig) -> None:
    """Walk ``cfg`` in place, remap any ``_target_`` string in ``_TARGET_REMAP``."""
    if not isinstance(cfg, DictConfig):
        return
    with OmegaConf.structured(cfg) if OmegaConf.is_struct(cfg) else _noop():
        OmegaConf.set_struct(cfg, False)
        _walk(cfg)


def _walk(node):
    from omegaconf import ListConfig

    if isinstance(node, DictConfig):
        if "_target_" in node and node["_target_"] in _TARGET_REMAP:
            node["_target_"] = _TARGET_REMAP[node["_target_"]]
        for v in node.values():
            _walk(v)
    elif isinstance(node, ListConfig):
        for v in node:
            _walk(v)


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _extract_submodule_weights(run_dir: str, ckpt_name: str):
    """Split a lightning ckpt into per-submodule .pt files (cached in ``{ckpt_name}_weights/``)."""
    import torch

    extracted = os.path.join(run_dir, f"{ckpt_name}_weights")
    os.makedirs(extracted, exist_ok=True)

    ckpt_dict = torch.load(os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt"))
    state_dict = ckpt_dict["state_dict"]

    module_names = {k.split(".")[0] for k in state_dict.keys()}
    for module_name in module_names:
        sub = {
            ".".join(k.split(".")[1:]): v.cpu()
            for k, v in state_dict.items()
            if k.split(".")[0] == module_name
        }
        torch.save(sub, os.path.join(extracted, f"{module_name}.pt"))


def load_model_from_cfg(cfg: DictConfig, ckpt_name: str = "last", device: str = "cpu", eval_mode: bool = True):
    import src.prepare  # noqa: F401  - logging / float32-matmul config
    import torch

    rewrite_targets(cfg)
    run_dir = cfg.run_dir

    model = hydra.utils.instantiate(cfg.model)

    pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    if not os.path.exists(pt_path):
        logger.info("Extracting %s checkpoint into per-submodule weights...", ckpt_name)
        _extract_submodule_weights(run_dir, ckpt_name)

    for fname in os.listdir(pt_path):
        module_name, ext = os.path.splitext(fname)
        if ext != ".pt":
            continue
        module = getattr(model, module_name, None)
        if module is None:
            continue
        state_dict = torch.load(os.path.join(pt_path, fname))
        module.load_state_dict(state_dict)
        logger.info("    %s loaded", module_name)

    model = model.to(device)
    if eval_mode:
        model = model.eval()
    return model


def load_model(run_dir: str, **params):
    cfg = read_config(run_dir)
    cfg.run_dir = run_dir
    return load_model_from_cfg(cfg, **params)
