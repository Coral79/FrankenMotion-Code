"""Shared inference helpers used by generate_part.py and eval_part.py.

These were inlined (and partially duplicated) in the two scripts. Extracted here
so both entry points share one implementation of model loading, joint extraction,
batch device movement, and the per-sample annotation JSON builder.
"""

from __future__ import annotations

import os
import random

import torch
from hydra.utils import instantiate


def load_diffusion(cfg, run_dir: str, ckpt_name: str, device: str):
    """Patch normalizer base_dirs at run_dir, instantiate, load ckpt, eval, to(device).

    Accepts either the training-time layout `run_dir/logs/checkpoints/<ckpt>.ckpt`
    or the released-HF flat layout `run_dir/<ckpt>.ckpt`.
    """
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(run_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(run_dir, "text_stats")

    diffusion = instantiate(cfg.diffusion)
    nested = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    flat = os.path.join(run_dir, f"{ckpt_name}.ckpt")
    ckpt_path = nested if os.path.exists(nested) else flat
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["state_dict"])
    diffusion.eval()
    diffusion.to(device)
    return diffusion


def load_smplh(gender: str = "male", jointstype: str = "both", path: str = "deps/smplh"):
    from src.tools.smpl_layer import SMPLH

    return SMPLH(
        path=path,
        jointstype=jointstype,
        input_pose_rep="axisangle",
        gender=gender,
    )


def extract_motion_outputs(xstart, length, nfeats, featsname, fps, value_from, smpl_layer):
    """Extract joints/vertices/smpldata from a diffusion output tensor."""
    from src.tools.extract_joints import extract_joints

    xstart_motion = xstart[:length, :nfeats]
    return extract_joints(
        xstart_motion,
        featsname,
        fps=fps,
        value_from=value_from,
        smpl_layer=smpl_layer,
    )


def resolve_clip_dir(cfg) -> str:
    """Locate the text-embedding directory for the configured text encoder.

    Released dataset layout: ``datasets/annotations/<dataname>/text_embeddings/<modelname>/``.
    Internal dataset layout: ``datasets/annotations/<dataname>/text_embeddings/<modelname>_embeddings/``.
    """
    text_emb_root = os.path.join(
        "datasets/annotations",
        cfg.data.text_encoder.dataname,
        "text_embeddings",
    )
    released = os.path.join(text_emb_root, cfg.data.text_encoder.modelname)
    internal = os.path.join(text_emb_root, f"{cfg.data.text_encoder.modelname}_embeddings")
    return released if os.path.isdir(released) else internal


def select_video_keyids(all_keyids, n: int, seed: int | None = None) -> set:
    """Deterministically sample ``n`` keyids for video rendering. Returns empty set if ``n<=0``."""
    if n <= 0 or not all_keyids:
        return set()
    rng = random.Random(seed) if seed is not None else random
    return set(rng.sample(list(all_keyids), min(n, len(all_keyids))))


def move_batch_to_device(batch: dict, device: str) -> dict:
    """In-place move tensors to ``device``. Handles one level of nested dicts (``tx``, ``tx_uncond_batch``)."""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            for k in batch[key]:
                if isinstance(batch[key][k], torch.Tensor):
                    batch[key][k] = batch[key][k].to(device)
    return batch


def build_annotation_json(
    text_input,
    split,
    dataset,
    key_id: str,
    length: int,
    global_text=None,
    fps: int = 20,
) -> dict:
    """Build the per-sample annotation dict saved next to each generated .npy."""
    annotation_data = {
        "sum_length": length,
        "caption": global_text if global_text else "",
    }

    if isinstance(text_input, str):
        body_part = "_".join(split.split("_")[1:]) if "_" in split else None

        if body_part == "total":
            try:
                body_part = dataset.annotations[key_id]["annotations"][0]["bodypart"]
            except (KeyError, IndexError):
                body_part = None

        if body_part == "sequence_caption":
            annotation_data["caption"] = text_input
        elif body_part:
            annotation_data[f"{body_part}_text"] = [text_input]
            annotation_data[f"{body_part}_length"] = [length]

    elif isinstance(text_input, dict):
        for part, part_data in text_input.items():
            if isinstance(part_data, list) and len(part_data) > 0:
                part_texts = [t for t, _ in part_data]
                part_lengths = [l for _, l in part_data]
                annotation_data[f"{part}_text"] = part_texts
                annotation_data[f"{part}_length"] = part_lengths

                if part == "merged_part_with_action":
                    _expand_merged_annotations(annotation_data, dataset, key_id, fps)
            else:
                annotation_data[f"{part}_text"] = ["unknown"]
                annotation_data[f"{part}_length"] = [length]

    return annotation_data


def _expand_merged_annotations(annotation_data: dict, dataset, key_id: str, fps: int):
    """Fill per-body-part text/length entries from raw dataset annotations."""
    body_parts = [
        "head", "left_arm", "left_leg",
        "right_arm", "right_leg", "spine",
        "trajectory", "action",
    ]
    try:
        annotations = dataset.annotations[key_id]["annotations"]
    except (KeyError, IndexError, TypeError):
        return

    for bodypart in body_parts:
        bp_anns = [
            ann for ann in annotations
            if isinstance(ann, dict) and ann.get("bodypart") == bodypart
        ]
        if not bp_anns:
            continue
        annotation_data[f"{bodypart}_text"] = [ann["text"] for ann in bp_anns]
        annotation_data[f"{bodypart}_length"] = [
            int(round((ann["end"] - ann["start"]) * fps)) for ann in bp_anns
        ]
