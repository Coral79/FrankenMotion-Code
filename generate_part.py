from __future__ import annotations
import logging
import os
import json
import shutil

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config import read_config
from src.tools.inference import (
    load_diffusion,
    load_smplh,
    extract_motion_outputs,
    resolve_clip_dir,
    move_batch_to_device,
)

# Avoid conflict between tokenizer threads and EGL rendering.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="generate_part", version_base="1.3")
def generate(c: DictConfig):
    logger.info("Prediction script")

    assert c.input_type in ("val", "user_input"), (
        f"Unsupported input_type {c.input_type!r}; expected 'val' or 'user_input'."
    )
    assert c.baseline in ["none", "sinc", "sinc_lerp"]

    import torch

    # `fast=true` clamps the whole run to 2 samples (skipping diffusion +
    # rendering for the rest), so the saved name reflects what was actually run.
    if c.input_type == "val":
        n_samples = c.num_sample if hasattr(c, 'num_sample') else 5
        if c.fast:
            n_samples = min(n_samples, 2)
        exp_folder_name = f"{c.input_type}_samples_{n_samples}"
    else:  # user_input
        exp_folder_name = os.path.splitext(os.path.basename(c.user_input_file))[0]

    if c.baseline != "none":
        exp_folder_name += "_baseline_" + c.baseline

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps
    if hasattr(c, 'condition'):
        cfg.data.condition = c.condition
    elif not hasattr(cfg.data, 'condition'):
        cfg.data.condition = 't2m'
    if hasattr(c, 'tiny'):
        cfg.data.tiny = c.tiny

    if c.input_type == "val":
        split = c.input_type
        dataset = instantiate(cfg.data, split=split)

        n_motions = c.num_sample if hasattr(c, 'num_sample') else min(5, len(dataset))
        if c.fast:
            n_motions = min(n_motions, 2)

        dataloader = instantiate(
            cfg.dataloader,
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            batch_size=n_motions,
        )

        batch = next(iter(dataloader))
        move_batch_to_device(batch, c.device)

        texts = batch.get('text', [])
        global_texts = batch.get('global_text', [None] * len(texts))
        lengths = batch['length'].tolist() if 'length' in batch else []

        infos = {
            "all_texts": texts,
            "all_lengths": lengths,
            "output_lengths": lengths,
            "split": split,
            "global_texts": global_texts,
        }

        logger.info(f"Loaded 1 batch from {split} dataset with {n_motions} samples")

    else:  # user_input
        from src.tools.parse_user_input import parse_and_validate_user_input
        from src.data.text_part_utils import load_from_annotation_with_model

        logger.info("Loading user input")

        annotation_dict = parse_and_validate_user_input(
            filepath=c.user_input_file,
            cfg=cfg,
            fps=fps,
        )

        text_encoder = instantiate(cfg.data.text_encoder)
        text_encoder.no_model = False  # allow on-the-fly embedding of user captions
        text_encoder.rand_mask = False  # disable training-time random masking at inference

        tx_emb_dict, text_dict = load_from_annotation_with_model(
            text_encoder=text_encoder,
            annotations=annotation_dict["annotations"],
            path=annotation_dict["path"],
            start=annotation_dict["start"],
            end=annotation_dict["end"],
        )

        duration_frames = int((annotation_dict["end"] - annotation_dict["start"]) * fps)
        motion_dim = cfg.data.motion_loader.nfeats
        text_dim = tx_emb_dict['local']["x"].shape[1]

        motion = torch.zeros((duration_frames, motion_dim))

        from src.data.text_motion import align
        text_aligned = align(motion, tx_emb_dict['local']["x"])
        x = torch.cat([motion, text_aligned], dim=1)

        batch = {
            "x": x.unsqueeze(0).to(c.device),
            "motion_dim": motion_dim,
            "text_dim": text_dim,
            "text": [text_dict],
            "tx": {
                "local": {
                    "x": tx_emb_dict['local']["x"].unsqueeze(0).to(c.device),
                    "mask": tx_emb_dict['local']["mask"].unsqueeze(0).to(c.device),
                    "length": tx_emb_dict['local']["length"],
                }
            },
            "length": torch.tensor([duration_frames]),
            "mask": torch.ones(1, duration_frames).bool(),
            "keyid": ["user_input_0"],
        }

        if "x" in tx_emb_dict:
            batch["tx"]["x"] = tx_emb_dict["x"].unsqueeze(0).to(c.device)
            batch["tx"]["length"] = torch.tensor([tx_emb_dict["length"]]).to(c.device)
            batch["global_text"] = [tx_emb_dict.get("text", "")]
        else:
            batch["global_text"] = [None]

        tx_uncond = text_encoder("unknown")
        batch["tx_uncond_batch"] = {
            "x": tx_uncond["x"].unsqueeze(0).to(c.device),
            "length": torch.tensor([tx_uncond["length"]]).to(c.device),
        }

        inpainting_motion_mask = torch.ones((duration_frames, motion_dim))
        inpainting_text_mask = torch.zeros((duration_frames, text_dim))
        batch["inpainting_mask"] = torch.cat(
            [inpainting_motion_mask, inpainting_text_mask], dim=1
        ).to(c.device)

        motion_mask = torch.ones_like(motion).to(c.device)
        text_mask_aligned = align(motion_mask, tx_emb_dict['local']["mask"]).to(c.device)
        batch["stats_mask"] = torch.cat(
            [motion_mask, text_mask_aligned], dim=1
        ).unsqueeze(0).to(c.device)

        sequence_caption = annotation_dict.get("sequence_caption", None)

        n_motions = 1
        split = None
        infos = {
            "all_texts": [text_dict],
            "all_lengths": [duration_frames],
            "output_lengths": [duration_frames],
            "global_texts": [sequence_caption],
        }

        logger.info(f"Loaded user input: {duration_frames} frames")

    logger.info("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np

    infos["featsname"] = cfg.motion_features
    infos["guidance_weight"] = c.guidance

    smpl_renderer = instantiate(c.smpl_renderer)

    logger.info("Loading the checkpoint and model")
    diffusion = load_diffusion(cfg, c.run_dir, c.ckpt, c.device)
    smplh = load_smplh(gender=c.gender)

    if hasattr(cfg.data, 'condition'):
        video_dir = os.path.join(
            c.run_dir,
            "generations",
            cfg.data.condition,
            exp_folder_name + "_" + str(c.ckpt) + f"_{c.input_type}_to_motion",
        )
    else:
        video_dir = os.path.join(
            c.run_dir,
            "generations",
            exp_folder_name + "_" + str(c.ckpt) + f"_{c.input_type}_to_motion",
        )
    os.makedirs(video_dir, exist_ok=True)

    if c.input_type == "val":
        info_file_path = os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}.txt")
        with open(info_file_path, 'w') as f:
            f.write(f"Input type: {c.input_type}\n")
            f.write(f"Number of samples: {n_motions}\n")
            if len(texts) > 0:
                f.write("\nTexts:\n")
                for i, text in enumerate(texts):
                    f.write(f"{i+1}. {text}\n")
    else:  # user_input
        input_ext = os.path.splitext(c.user_input_file)[1]
        shutil.copy(
            c.user_input_file,
            os.path.join(video_dir, f"{exp_folder_name}_user_input{input_ext}"),
        )

    smpl_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_smpl.mp4")
        for idx in range(n_motions)
    ]

    npy_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}.npy")
        for idx in range(n_motions)
    ]

    logger.info(f"All the output videos will be saved in: {video_dir}")

    if c.seed != -1:
        pl.seed_everything(c.seed)

    with torch.no_grad():
        logger.info(
            f"Processing {'user input' if c.input_type == 'user_input' else split}"
            f" with {n_motions} sample(s)"
        )
        xstarts = diffusion.batch_forward(batch, infos).cpu()

        for idx, (xstart, length) in enumerate(zip(xstarts, infos["output_lengths"])):
            text_input = infos["all_texts"][idx] if idx < len(infos["all_texts"]) else None
            global_text_input = infos["global_texts"][idx] if idx < len(infos["global_texts"]) else None

            output = extract_motion_outputs(
                xstart,
                length,
                cfg.data.motion_loader.nfeats,
                infos["featsname"],
                fps,
                c.value_from,
                smplh,
            )

            np.save(npy_paths[idx], output["joints"])

            # Save the input (ground-truth) text that conditioned the model — useful
            # for visual inspection of how the generated motion aligns with the prompt.
            input_text_path = npy_paths[idx].replace(".npy", "_input_text.txt")
            with open(input_text_path, "w") as f:
                if global_text_input:
                    f.write(f"sequence_caption: {global_text_input}\n\n")
                if isinstance(text_input, dict):
                    for part_name, segments in text_input.items():
                        # Skip the `<part>_length` keys carried by the val-mode dict.
                        if part_name.endswith("_length"):
                            continue
                        if not isinstance(segments, list):
                            continue
                        f.write(f"{part_name}:\n")
                        for seg in segments:
                            # Segment may be (text, length) tuple, {"text", "start", "end"} dict, or raw str.
                            if isinstance(seg, tuple) and len(seg) == 2:
                                seg_text, seg_length = seg
                                f.write(f"  - [{seg_length} frames] {seg_text}\n")
                            elif isinstance(seg, dict) and "text" in seg:
                                start, end = seg.get("start"), seg.get("end")
                                if start is not None and end is not None:
                                    f.write(f"  - [{start:.1f}s–{end:.1f}s] {seg['text']}\n")
                                else:
                                    f.write(f"  - {seg['text']}\n")
                            else:
                                f.write(f"  - {seg}\n")
                elif text_input is not None:
                    f.write(str(text_input))

            if "vertices" in output:
                np.save(npy_paths[idx].replace(".npy", "_verts.npy"), output["vertices"])

            if "smpldata" in output:
                np.savez(npy_paths[idx].replace(".npy", "_smpl.npz"), **output["smpldata"])

            if "vertices" in output:
                logger.info(f"SMPL rendering {idx}")
                if cfg.data.condition == "t2m":
                    smpl_renderer(
                        output["vertices"],
                        title=global_text_input,
                        output=smpl_video_paths[idx],
                        annotations=text_input,
                    )
                else:
                    smpl_renderer(
                        output["vertices"],
                        title=global_text_input,
                        output=smpl_video_paths[idx],
                        annotations=ann_list_dict,
                    )
                logger.info(smpl_video_paths[idx])

            logger.info("Rendering done")

        # For val, also render the ground truth alongside the prediction.
        if c.input_type == "val":
            for idx, (xstart, length, text_input, global_text_input) in enumerate(
                zip(batch['x'].cpu(), infos["output_lengths"], infos["all_texts"], infos["global_texts"])
            ):
                output = extract_motion_outputs(
                    xstart,
                    length,
                    cfg.data.motion_loader.nfeats,
                    infos["featsname"],
                    fps,
                    c.value_from,
                    smplh,
                )

                np.save(npy_paths[idx][:-4] + '_gt.npy', output["joints"])

                if "vertices" in output:
                    np.save(npy_paths[idx].replace(".npy", "_verts_gt.npy"), output["vertices"])

                if "smpldata" in output:
                    np.savez(npy_paths[idx].replace(".npy", "_smpl_gt.npz"), **output["smpldata"])

                if "vertices" in output:
                    logger.info(f"SMPL rendering {idx}")
                    smpl_renderer(
                        output["vertices"],
                        title=global_text_input,
                        output=smpl_video_paths[idx][:-4] + '_gt.mp4',
                        annotations=text_input,
                    )
                    logger.info(smpl_video_paths[idx][:-4] + '_gt.mp4')

                logger.info("Rendering done")


if __name__ == "__main__":
    generate()
