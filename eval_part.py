"""Evaluation generation: runs the full val/test split ``num_runs`` times and writes
a structured output tree consumed by ``src.eval.run``.

Output layout::

    {run_dir}/generations/{condition}/{split}_{ckpt}_{timestamp}/
        run_summary.json
        video_samples.json
        gt/{keyid}.npy               # dict with 'joints', 'smplrifke', 'smpldata'?
        gt/{keyid}.json              # annotation dict (sum_length, caption, <part>_text, <part>_length, ...)
        turn_0/data/{keyid}.npy
        turn_0/data/{keyid}.json
        turn_0/videos/{keyid}_joints.mp4     # only for selected video_key_ids (first turn only)
        turn_0/videos/{keyid}_smpl.mp4
        turn_1/data/...
        ...
"""

from __future__ import annotations

import datetime
import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from src.config import read_config
from src.tools.inference import (
    build_annotation_json,
    extract_motion_outputs,
    load_diffusion,
    load_smplh,
    move_batch_to_device,
    select_video_keyids,
)

# Avoid conflict between tokenizer threads and EGL rendering.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="eval_part", version_base="1.3")
def generate(c: DictConfig):
    logger.info("Eval generation: full %s split × %d runs", c.input_type, c.num_runs)

    assert "val" in c.input_type or "test" in c.input_type, (
        f"eval_part expects a val/test split, got {c.input_type!r}"
    )
    assert c.baseline in ["none", "sinc", "sinc_lerp"]

    import torch

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_runs = int(c.num_runs)
    video_samples_per_run = int(c.video_samples_per_run)

    exp_folder_name = f"{c.input_type}"
    if c.baseline != "none":
        exp_folder_name += "_baseline_" + c.baseline

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps

    if hasattr(c, "condition"):
        cfg.data.condition = c.condition
    elif not hasattr(cfg.data, "condition"):
        cfg.data.condition = "t2m"
    if hasattr(c, "tiny"):
        cfg.data.tiny = c.tiny

    split = c.input_type
    dataset = instantiate(cfg.data, split=split)

    dataloader = instantiate(
        cfg.dataloader,
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    total_samples = len(dataset)
    logger.info(
        "Split=%s; %d samples; batch_size=%d; num_workers=%d",
        split, total_samples, cfg.dataloader.batch_size, cfg.dataloader.num_workers,
    )

    logger.info("Loading libraries")
    import src.prepare  # noqa: F401
    import pytorch_lightning as pl
    import numpy as np

    logger.info("Loading checkpoint + diffusion model")
    diffusion = load_diffusion(cfg, c.run_dir, c.ckpt, c.device)
    smplh = load_smplh(gender=c.gender)

    joints_renderer = instantiate(c.joints_renderer) if video_samples_per_run > 0 else None
    smpl_renderer = instantiate(c.smpl_renderer) if video_samples_per_run > 0 else None

    base_output_dir = os.path.join(
        c.run_dir,
        "generations",
        cfg.data.condition,
        f"{exp_folder_name}_{c.ckpt}_{timestamp}",
    )
    gt_dir = os.path.join(base_output_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    # Gather all key_ids up front so the video-sample selection is stable across runs.
    all_key_ids: list[str] = []
    for batch in dataloader:
        all_key_ids.extend(batch["keyid"])

    video_key_ids = select_video_keyids(all_key_ids, video_samples_per_run, seed=c.seed if c.seed != -1 else None)

    with open(os.path.join(base_output_dir, "video_samples.json"), "w") as f:
        json.dump(
            {"video_key_ids": sorted(video_key_ids), "total_samples": len(all_key_ids)},
            f, indent=2,
        )
    logger.info(
        "Video samples: %d of %d (%s)",
        len(video_key_ids), len(all_key_ids), sorted(video_key_ids),
    )

    # Save GT once + cache CPU-side batches for reuse across runs
    logger.info("Saving ground truth...")
    batch_cache: list[dict] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="GT")):
            move_batch_to_device(batch, c.device)
            batch_cache.append({
                "batch": {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                },
                "batch_idx": batch_idx,
            })

            key_ids = batch["keyid"]
            texts = batch.get("text", [None] * len(key_ids))
            global_texts = batch.get("global_text", [None] * len(key_ids))
            lengths = batch["length"].tolist()

            for gt_motion, length, text_input, global_text_input, key_id in zip(
                batch["x"].cpu(), lengths, texts, global_texts, key_ids,
            ):
                gt_output = extract_motion_outputs(
                    gt_motion, length, cfg.data.motion_loader.nfeats,
                    cfg.motion_features, fps, c.value_from, smplh,
                )
                gt_dict = {
                    "joints": gt_output["joints"],
                    "smplrifke": gt_motion[:length, :cfg.data.motion_loader.nfeats].cpu().numpy(),
                }
                if "smpldata" in gt_output:
                    gt_dict["smpldata"] = gt_output["smpldata"]
                np.save(os.path.join(gt_dir, f"{key_id}.npy"), gt_dict)

                ann = build_annotation_json(
                    text_input=text_input,
                    split=split,
                    dataset=dataset,
                    key_id=key_id,
                    length=length,
                    global_text=global_text_input,
                    fps=fps,
                )
                with open(os.path.join(gt_dir, f"{key_id}.json"), "w") as f:
                    json.dump(ann, f, indent=2)

    logger.info("Ground truth saved")

    gt_videos_rendered = False

    for run_idx in range(num_runs):
        logger.info("Run %d/%d", run_idx + 1, num_runs)

        run_out = os.path.join(base_output_dir, f"turn_{run_idx}")
        data_dir = os.path.join(run_out, "data")
        os.makedirs(data_dir, exist_ok=True)

        video_dir = None
        if video_samples_per_run > 0:
            video_dir = os.path.join(run_out, "videos")
            os.makedirs(video_dir, exist_ok=True)

            if run_idx == 0 and not gt_videos_rendered:
                logger.info("Rendering GT videos for selected samples...")
                _render_gt_videos(
                    batch_cache=batch_cache,
                    video_key_ids=video_key_ids,
                    dataset=dataset,
                    cfg=cfg,
                    fps=fps,
                    value_from=c.value_from,
                    smplh=smplh,
                    joints_renderer=joints_renderer,
                    smpl_renderer=smpl_renderer,
                    video_dir=video_dir,
                    device=c.device,
                    render_smpl=not c.fast,
                )
                gt_videos_rendered = True
                logger.info("GT videos done")

        if c.seed != -1:
            pl.seed_everything(c.seed + run_idx * 1000)

        with torch.no_grad():
            for cached in tqdm(batch_cache, desc=f"Run {run_idx + 1}"):
                batch = cached["batch"]
                move_batch_to_device(batch, c.device)

                key_ids = batch["keyid"]
                texts = batch.get("text", [None] * len(key_ids))
                global_texts = batch.get("global_text", [None] * len(key_ids))
                lengths = batch["length"].tolist()

                infos = {
                    "all_texts": texts,
                    "all_lengths": lengths,
                    "output_lengths": lengths,
                    "split": split,
                    "global_texts": global_texts,
                    "featsname": cfg.motion_features,
                    "guidance_weight": c.guidance,
                }

                xstarts = diffusion.batch_forward(batch, infos).cpu()

                for xstart, length, text_input, global_text_input, key_id in zip(
                    xstarts, lengths, texts, global_texts, key_ids,
                ):
                    output = extract_motion_outputs(
                        xstart, length, cfg.data.motion_loader.nfeats,
                        infos["featsname"], fps, c.value_from, smplh,
                    )
                    gen_dict = {
                        "joints": output["joints"],
                        "smplrifke": xstart.cpu().numpy(),
                    }
                    if "smpldata" in output:
                        gen_dict["smpldata"] = output["smpldata"]
                    np.save(os.path.join(data_dir, f"{key_id}.npy"), gen_dict)

                    ann = build_annotation_json(
                        text_input=text_input,
                        split=split,
                        dataset=dataset,
                        key_id=key_id,
                        length=length,
                        global_text=global_text_input,
                        fps=fps,
                    )
                    with open(os.path.join(data_dir, f"{key_id}.json"), "w") as f:
                        json.dump(ann, f, indent=2)

                    if run_idx == 0 and key_id in video_key_ids and video_dir is not None:
                        _render_pred_video(
                            key_id=key_id,
                            joints=output["joints"],
                            output=output,
                            text_input=text_input,
                            global_text_input=global_text_input,
                            dataset=dataset,
                            fps=fps,
                            joints_renderer=joints_renderer,
                            smpl_renderer=smpl_renderer,
                            video_dir=video_dir,
                            render_smpl=not c.fast,
                        )

        logger.info("Run %d done", run_idx + 1)

    with open(os.path.join(base_output_dir, "run_summary.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_runs": num_runs,
            "total_samples": len(all_key_ids),
            "video_samples_per_run": video_samples_per_run,
            "config": {
                "batch_size": cfg.dataloader.batch_size,
                "num_workers": cfg.dataloader.num_workers,
                "split": split,
                "ckpt": c.ckpt,
            },
        }, f, indent=2)

    logger.info("All runs complete. Output: %s", base_output_dir)


# ---- helpers kept file-local since they only show up in eval-generation ----

def _video_annotations(text_input, dataset, key_id, fps):
    """Build the per-part annotation dict consumed by the SMPL renderer's
    on-screen overlay. Expands ``merged_part_with_action`` the same way
    ``build_annotation_json`` does.
    """
    if not isinstance(text_input, dict):
        return None

    if "merged_part_with_action" not in text_input:
        return text_input

    try:
        annotations = dataset.annotations[key_id]["annotations"]
    except (KeyError, IndexError, TypeError):
        return text_input

    body_parts = [
        "head", "left_arm", "left_leg",
        "right_arm", "right_leg", "spine",
        "trajectory", "action",
    ]
    out = {}
    for bp in body_parts:
        bp_anns = [a for a in annotations if isinstance(a, dict) and a.get("bodypart") == bp]
        if bp_anns:
            out[bp] = [
                (a["text"], int(round((a["end"] - a["start"]) * fps)))
                for a in bp_anns
            ]
    return out or text_input


def _render_gt_videos(
    *, batch_cache, video_key_ids, dataset, cfg, fps, value_from,
    smplh, joints_renderer, smpl_renderer, video_dir, device, render_smpl,
):
    for cached in batch_cache:
        batch = cached["batch"]
        move_batch_to_device(batch, device)

        key_ids = batch["keyid"]
        texts = batch.get("text", [None] * len(key_ids))
        global_texts = batch.get("global_text", [None] * len(key_ids))
        lengths = batch["length"].tolist()

        for gt_motion, length, text_input, global_text_input, key_id in zip(
            batch["x"], lengths, texts, global_texts, key_ids,
        ):
            if key_id not in video_key_ids:
                continue

            gt_output = extract_motion_outputs(
                gt_motion.cpu(), length, cfg.data.motion_loader.nfeats,
                cfg.motion_features, fps, value_from, smplh,
            )
            display = global_text_input or (text_input if isinstance(text_input, str) else "")

            joints_renderer(
                gt_output["joints"],
                title=f"GT - {display}",
                output=os.path.join(video_dir, f"gt_{key_id}_joints.mp4"),
                canonicalize=False,
            )

            if "vertices" in gt_output and render_smpl:
                smpl_renderer(
                    gt_output["vertices"],
                    title=f"GT - {display}",
                    output=os.path.join(video_dir, f"gt_{key_id}_smpl.mp4"),
                    annotations=_video_annotations(text_input, dataset, key_id, fps),
                )


def _render_pred_video(
    *, key_id, joints, output, text_input, global_text_input,
    dataset, fps, joints_renderer, smpl_renderer, video_dir, render_smpl,
):
    display = global_text_input or (text_input if isinstance(text_input, str) else "")

    joints_renderer(
        joints,
        title=display,
        output=os.path.join(video_dir, f"{key_id}_joints.mp4"),
        canonicalize=False,
    )

    if "vertices" in output and render_smpl:
        smpl_renderer(
            output["vertices"],
            title=display,
            output=os.path.join(video_dir, f"{key_id}_smpl.mp4"),
            annotations=_video_annotations(text_input, dataset, key_id, fps),
        )


if __name__ == "__main__":
    generate()
