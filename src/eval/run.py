import os
import numpy as np
import torch
import argparse
from pathlib import Path

import src.prepare  # noqa
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Force true fp32 on cuda — must come AFTER src.prepare, which sets
# torch.set_float32_matmul_precision("high") and would re-enable tf32 otherwise.
# Without this, cuDNN tf32 flips near-tied top-1 sims and bare-guo Per-action /
# Per-parts R@1 drift ~3-4pt vs cpu.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from src.eval.metrics.metrics import (
    calculate_activation_statistics,
    calculate_activation_statistics_normalized,
    calculate_frechet_distance,
    calculate_diversity,
    calculate_matching_score,
)
from src.eval.eval_model.load import load_eval_model
from src.eval.load_generated_data import (
    load_experiment_data_enhanced,
    load_run_summary,
    BODY_PARTS,
)
from src.tools.guofeats import joints_to_guofeats, guofeats_to_joints_local
from src.eval.eval_model.model import get_sim_matrix
from src.eval.metrics.contrastive import all_contrastive_metrics

def get_ordered_metrics_columns(sample_metrics=None):
    """Define metric order and formatting. Core metrics first."""
    core = [
        ("fid", "FID", ".3f"), ("fid+", "FID+", ".3f"), ("fid_nrom", "FID_nrom", ".3f"),
        ("mm_dist", "MM_Dist", ".3f"), ("diversity", "Diversity", ".3f"),
    ]
    retrieval = [
        ("m2t_top_1", "R1", ".3f"), ("m2t_top_3", "R3", ".3f"), ("m2t_top_10", "R10", ".3f"),
    ]
    similarity = [("m2t_score", "M2T", ".3f"), ("m2m_score", "M2M", ".3f")]
    physical = [("transition", "Trans", ".3f")]
    m2t_contra = [
        ("m2t/R01", "m2t_R01", ".3f"), ("m2t/R03", "m2t_R03", ".3f"),
        ("m2t/R05", "m2t_R05", ".3f"), ("m2t/R10", "m2t_R10", ".3f"),
        ("m2t/MedR", "m2t_MedR", ".3f"), ("m2t/MeanR", "m2t_MeanR", ".3f"),
    ]
    t2m_contra = [
        ("t2m/R01", "t2m_R01", ".3f"), ("t2m/R03", "t2m_R03", ".3f"),
        ("t2m/R05", "t2m_R05", ".3f"), ("t2m/R10", "t2m_R10", ".3f"),
        ("t2m/MedR", "t2m_MedR", ".3f"), ("t2m/MeanR", "t2m_MeanR", ".3f"),
    ]
    
    ordered = core + retrieval + similarity + physical + m2t_contra + t2m_contra
    
    if sample_metrics is not None:
        available = set(sample_metrics.keys())
        ordered = [(k, n, f) for k, n, f in ordered if k in available]
        predefined = {k for k, _, _ in ordered}
        for key in sample_metrics.keys():
            if key not in predefined:
                val = sample_metrics[key]
                fmt = ".3f" if isinstance(val, (float, np.floating)) else ""
                ordered.append((key, key, fmt))
    
    return ordered


def format_metrics_for_csv(metrics_dict):
    """Format metrics into OrderedDict with preferred ordering."""
    from collections import OrderedDict
    ordered_metrics = OrderedDict()
    metric_spec = get_ordered_metrics_columns(sample_metrics=metrics_dict)
    
    for metric_key, display_name, format_spec in metric_spec:
        if metric_key in metrics_dict:
            value = metrics_dict[metric_key]
            try:
                formatted_value = float(format(value, format_spec)) if format_spec else value
            except (ValueError, TypeError):
                formatted_value = value
            ordered_metrics[display_name] = formatted_value
    
    return ordered_metrics


def save_results_to_csv(results, save_path, args):
    """Save evaluation results to CSV files."""
    import pandas as pd
    from collections import OrderedDict
    
    save_dir = os.path.dirname(save_path) or "."
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    
    # Caption CSV
    if "caption" in results["gt_metrics"]:
        caption_data = []
        gt_metrics = format_metrics_for_csv(results["gt_metrics"]["caption"])
        caption_data.append(OrderedDict([("run", "GT")] + list(gt_metrics.items())))
        
        for run_idx in range(len(results["run_metrics"].get("caption", []))):
            run_metrics = format_metrics_for_csv(results["run_metrics"]["caption"][run_idx])
            caption_data.append(OrderedDict([("run", f"run_{run_idx:03d}")] + list(run_metrics.items())))
        
        if results["avg_metrics"].get("caption"):
            avg_metrics = format_metrics_for_csv(results["avg_metrics"]["caption"])
            caption_data.append(OrderedDict([("run", "AVERAGE")] + list(avg_metrics.items())))
            std_metrics = format_metrics_for_csv(results["std_metrics"]["caption"])
            caption_data.append(OrderedDict([("run", "STD")] + list(std_metrics.items())))
        
        pd.DataFrame(caption_data).to_csv(f"{save_dir}/{base_name}_caption.csv", index=False)
    
    # Action CSV
    if "action" in results["gt_metrics"]:
        action_data = []
        gt_metrics = format_metrics_for_csv(results["gt_metrics"]["action"])
        action_data.append(OrderedDict([("run", "GT")] + list(gt_metrics.items())))
        
        for run_idx in range(len(results["run_metrics"].get("action", []))):
            run_metrics = format_metrics_for_csv(results["run_metrics"]["action"][run_idx])
            action_data.append(OrderedDict([("run", f"run_{run_idx:03d}")] + list(run_metrics.items())))
        
        if results["avg_metrics"].get("action"):
            avg_metrics = format_metrics_for_csv(results["avg_metrics"]["action"])
            action_data.append(OrderedDict([("run", "AVERAGE")] + list(avg_metrics.items())))
            std_metrics = format_metrics_for_csv(results["std_metrics"]["action"])
            action_data.append(OrderedDict([("run", "STD")] + list(std_metrics.items())))
        
        pd.DataFrame(action_data).to_csv(f"{save_dir}/{base_name}_action.csv", index=False)
    
    # Parts CSV
    parts_data = []
    gt_row = OrderedDict([("run", "GT")])
    for part in BODY_PARTS:
        if part in results["gt_metrics"].get("parts", {}):
            part_metrics = format_metrics_for_csv(results["gt_metrics"]["parts"][part])
            for metric_name, metric_value in part_metrics.items():
                gt_row[f"{part}_{metric_name}"] = metric_value
    parts_data.append(gt_row)
    
    for run_idx in range(results['num_runs']):
        run_row = OrderedDict([("run", f"run_{run_idx:03d}")])
        for part in BODY_PARTS:
            if part in results["run_metrics"]["parts"] and run_idx < len(results["run_metrics"]["parts"][part]):
                part_metrics = format_metrics_for_csv(results["run_metrics"]["parts"][part][run_idx])
                for metric_name, metric_value in part_metrics.items():
                    run_row[f"{part}_{metric_name}"] = metric_value
        parts_data.append(run_row)
    
    avg_row = OrderedDict([("run", "AVERAGE")])
    std_row = OrderedDict([("run", "STD")])
    for part in BODY_PARTS:
        if part in results["gt_metrics"].get("parts", {}):
            part_avg_metrics = {}
            part_std_metrics = {}
            for metric_name in results["gt_metrics"]["parts"][part].keys():
                values = [results["run_metrics"]["parts"][part][run_idx][metric_name]
                         for run_idx in range(len(results["run_metrics"]["parts"].get(part, [])))]
                part_avg_metrics[metric_name] = np.mean(values) if values else 0
                part_std_metrics[metric_name] = np.std(values) if values else 0
            formatted_avg = format_metrics_for_csv(part_avg_metrics)
            formatted_std = format_metrics_for_csv(part_std_metrics)
            for metric_name, metric_value in formatted_avg.items():
                avg_row[f"{part}_{metric_name}"] = metric_value
            for metric_name, metric_value in formatted_std.items():
                std_row[f"{part}_{metric_name}"] = metric_value
    parts_data.append(avg_row)
    parts_data.append(std_row)
    
    pd.DataFrame(parts_data).to_csv(f"{save_dir}/{base_name}_parts.csv", index=False)
    
    # Parts Average CSV
    parts_avg_data = []
    gt_metrics = format_metrics_for_csv(results["gt_metrics"].get("parts_avg", {}))
    parts_avg_data.append(OrderedDict([("run", "GT")] + list(gt_metrics.items())))
    
    for run_idx in range(results['num_runs']):
        run_metrics_raw = {}
        for metric_name in results["avg_metrics"].get("parts_avg", {}).keys():
            values = [results["run_metrics"]["parts"][part][run_idx][metric_name]
                     for part in BODY_PARTS
                     if part in results["run_metrics"]["parts"] and run_idx < len(results["run_metrics"]["parts"][part])]
            run_metrics_raw[metric_name] = np.mean(values) if values else 0
        run_metrics = format_metrics_for_csv(run_metrics_raw)
        parts_avg_data.append(OrderedDict([("run", f"run_{run_idx:03d}")] + list(run_metrics.items())))
    
    if results["avg_metrics"].get("parts_avg"):
        avg_metrics = format_metrics_for_csv(results["avg_metrics"]["parts_avg"])
        parts_avg_data.append(OrderedDict([("run", "AVERAGE")] + list(avg_metrics.items())))
        std_metrics = format_metrics_for_csv(results["std_metrics"]["parts_avg"])
        parts_avg_data.append(OrderedDict([("run", "STD")] + list(std_metrics.items())))
    
    pd.DataFrame(parts_avg_data).to_csv(f"{save_dir}/{base_name}_parts_average.csv", index=False)
    
    # Metadata CSV
    metadata = {
        "experiment_dir": args.experiment_dir,
        "dataset_name": args.dataset_name,
        "protocol": args.protocol,
        "num_runs_evaluated": results['num_runs'],
    }
    pd.DataFrame([metadata]).to_csv(f"{save_dir}/{base_name}_metadata.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {save_dir}/{base_name}_caption.csv")
    print(f"  - {save_dir}/{base_name}_action.csv")
    print(f"  - {save_dir}/{base_name}_parts.csv")
    print(f"  - {save_dir}/{base_name}_parts_average.csv")
    print(f"  - {save_dir}/{base_name}_metadata.csv")


def format_metrics_for_csv(metrics_dict):
    """Format metrics into OrderedDict with preferred ordering."""
    from collections import OrderedDict
    ordered_metrics = OrderedDict()
    metric_spec = get_ordered_metrics_columns(sample_metrics=metrics_dict)
    
    for metric_key, display_name, format_spec in metric_spec:
        if metric_key in metrics_dict:
            value = metrics_dict[metric_key]
            try:
                formatted_value = float(format(value, format_spec)) if format_spec else value
            except (ValueError, TypeError):
                formatted_value = value
            ordered_metrics[display_name] = formatted_value
    
    return ordered_metrics

def get_dataset_name_from_splits(split_path):
    """Infer dataset name from split_path."""
    if split_path is None:
        return "babel-part"  # default
    parent_dir = os.path.basename(os.path.dirname(split_path))
    return parent_dir


def get_model_dirs_for_dataset(dataset_name, base_eval_model_dir):
    """Build MODEL_DIRS based on dataset name.

    Prefers the release layout ``<base>/<part>/`` (``<base>/head/``,
    ``<base>/action/``, ``<base>/sequence_caption/``) and falls back to the
    legacy tmr_plus layout ``<base>/tmr_<dataset>_guoh3dfeats_<part>/`` so the
    same code can evaluate release-era and stmc-era generation folders.
    """
    model_dirs = {}

    def _pick(part_or_key):
        released = os.path.join(base_eval_model_dir, part_or_key)
        legacy = os.path.join(base_eval_model_dir, f"tmr_{dataset_name}_guoh3dfeats_{part_or_key}")
        return released if os.path.exists(released) else legacy

    for part in ["head", "left_arm", "right_arm", "left_leg", "right_leg", "spine", "trajectory"]:
        model_dirs[part] = _pick(part)

    model_dirs["action"] = _pick("action")
    caption_path = _pick("sequence_caption")
    model_dirs["caption"] = caption_path
    model_dirs["sequence_caption"] = caption_path

    return model_dirs


def load_eval_model_for_dataset(dataset_type, model_dirs, default_model_dir, device):
    """Load the retrieval eval model for a dataset type, falling back to the default."""
    model_dir = model_dirs.get(dataset_type, default_model_dir)

    if not os.path.exists(model_dir):
        print(f"Warning: Model not found at {model_dir}, using default")
        model_dir = default_model_dir

    return load_eval_model(device=device, run_dir=model_dir)


def print_latex(name, metrics):
    latex = [
        name.ljust(32),
        "{:.3f}".format(metrics["m2t/R01"]).ljust(4),
        "{:.3f}".format(metrics["m2t/R03"]).ljust(4),
        "{:.3f}".format(metrics["m2t_score"]).ljust(5),
        "{:.3f}".format(metrics["mm_dist"]).ljust(5),
        "{:.3f}".format(metrics["diversity"]).ljust(5),
        "{:.3f}".format(metrics["fid"]).ljust(5),
        "{:.3f}".format(metrics["fid+"]).ljust(5),
        "{:.3f}".format(metrics["fid_nrom"]).ljust(5),
        "{:.3f}".format(100 * metrics["transition"]).ljust(3),
    ]
    line = " & ".join(latex) + r" \\"
    print(line)
    return

def calculate_fid_realism(gfeats, tmr_forward, idx_seed, fps=20.0, nb_samples=5):
    """Calculate FID realism score using 5 random 5-second crops."""
    m_len = gfeats.shape[0]
    n_realism_seq = 5.0
    n_real_nframes = int(n_realism_seq * fps)
    
    if m_len > n_real_nframes:
        np.random.seed(idx_seed)
        realism_idx = np.random.randint(0, m_len - n_real_nframes, nb_samples)
        realism_crop_motions = [gfeats[x:x + n_real_nframes] for x in realism_idx]
    else:
        realism_crop_motions = [gfeats] * nb_samples
    
    return tmr_forward(realism_crop_motions)


def calculate_transition_distance(joints_local):
    """Calculate transition distance at 1/4, 1/2, 3/4 points."""
    N = len(joints_local)
    if N < 4:
        return None
    
    inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4])
    return torch.linalg.norm(
        (joints_local[inter_points] - joints_local[inter_points - 1]),
        dim=-1,
    ).mean(-1).flatten()


def calculate_m2t_scores_batch(sim_matrix_m2t, text_indices=None):
    """Calculate M2T scores. For GT: text_indices=None (uses diagonal). For Gen: pass text_indices."""
    if text_indices is None:
        text_indices = range(len(sim_matrix_m2t))
    
    scores = [(sim_matrix_m2t[i, text_idx] + 1) / 2 
              for i, text_idx in enumerate(text_indices)]
    return float(np.mean(scores))


def calculate_m2m_scores_batch(sim_matrix_m2m, motion_indices=None):
    """Calculate M2M scores. For GT: motion_indices=None (uses diagonal). For Gen: pass motion_indices."""
    if motion_indices is None:
        motion_indices = range(len(sim_matrix_m2m))
    
    scores = [(sim_matrix_m2m[i, motion_idx] + 1) / 2 
              for i, motion_idx in enumerate(motion_indices)]
    return float(np.mean(scores))


def calculate_m2t_top_k_batch(sim_matrix_m2t, text_indices=None, k=1):
    """Calculate M2T top-k. For GT: text_indices=None (uses diagonal). For Gen: pass text_indices."""
    if text_indices is None:
        text_indices = range(len(sim_matrix_m2t))
    
    top_k_lst = []
    for i, text_idx in enumerate(text_indices):
        asort = np.argsort(sim_matrix_m2t[i])[::-1]
        top_k_lst.append(1 if text_idx in asort[:k] else 0)
    return float(np.mean(top_k_lst))


def get_gt_metrics_for_dataset(texts, motions_guofeats, tmr_forward, 
                               sent_emb=None, protocol="normal", batch_size=256):
    """
    Calculate GT metrics for a single dataset, including contrastive retrieval metrics.
    """
    if sent_emb is not None and len(sent_emb) > 0:
        sent_emb = np.stack(sent_emb)
    else:
        sent_emb = None  # downstream indexing expects None-or-ndarray, not []
    # sent_emb = torch.from_numpy(sent_emb)
    # Base metrics (by definition for GT)
    metrics = {"m2m_score": 1.00, "fid+": 0.00}

    # Compute similarity matrix
    sim_matrix, motion_latents_gt, text_latents_gt = compute_sim_matrix_m2t(motions_guofeats, texts, tmr_forward, batch_size)

    # Legacy motion-to-text retrieval metrics for quick compatibility
    metrics["m2t_score"] = calculate_m2t_scores_batch(sim_matrix)
    metrics["m2t_top_1"] = calculate_m2t_top_k_batch(sim_matrix, k=1)
    metrics["m2t_top_3"] = calculate_m2t_top_k_batch(sim_matrix, k=3)

    # Transition distance
    trans_dist_lst = []
    for motion_guofeats in motions_guofeats:
        motion_tensor = torch.from_numpy(motion_guofeats)
        joints_local = guofeats_to_joints_local(motion_tensor)
        joints_local = joints_local - joints_local[:, [0]]
        
        trans_dist = calculate_transition_distance(joints_local)
        if trans_dist is not None:
            trans_dist_lst.append(trans_dist)

    metrics["transition"] = (
        float(torch.concatenate(trans_dist_lst).mean()) if trans_dist_lst else 0.0
    )
    
    nb_sample = sim_matrix.shape[0]
    diversity = calculate_diversity(motion_latents_gt, 300 if nb_sample > 300 else 50)
    metrics["diversity"]=diversity
    
    # === MM-DIST (Matching Score) ===
    mm_dist_all = calculate_matching_score(text_latents_gt, motion_latents_gt, sum_all=False)
    metrics["mm_dist"] = float(mm_dist_all.mean())   # average distance per dataset
    
    gt_mu, gt_cov = calculate_activation_statistics(motion_latents_gt.detach().cpu().numpy())
    fid = calculate_frechet_distance(gt_mu, gt_cov, gt_mu, gt_cov)
    metrics["fid"]=fid 
    
    gt_mu_nrom, gt_cov_nrom = calculate_activation_statistics(motion_latents_gt.detach().cpu().numpy())
    fid_nrom = calculate_frechet_distance(gt_mu_nrom, gt_cov_nrom, gt_mu_nrom, gt_cov_nrom)
    metrics["fid_nrom"]=fid_nrom
    
    
    # Protocol-specific contrastive metrics
    contrastive = compute_contrastive_metrics_by_protocol(
        sim_matrix, sent_emb, protocol
    )
    metrics.update(contrastive)


        
    return metrics


def calculate_dataset_metrics(texts, motions_guofeats, tmr_forward, 
                             motions_guofeats_gt,
                             fps=20.0, sent_emb=None, protocol="normal", batch_size=256):
    """Calculate metrics for a single dataset."""
    
    ### SEMANTICS (BATCHED) ###

    if sent_emb is not None and len(sent_emb) > 0:
        sent_emb = np.stack(sent_emb)
    else:
        sent_emb = None  # downstream indexing expects None-or-ndarray, not []
        
    ### SEMANTICS ###
    sim_matrix_m2t, motion_latents, text_latents = compute_sim_matrix_m2t(motions_guofeats, texts, tmr_forward, batch_size)
    sim_matrix_m2m, _, motion_latents_gt = compute_sim_matrix_m2m(motions_guofeats, motions_guofeats_gt, tmr_forward, batch_size)
    
    metrics = {}
    metrics["m2t_score"] = calculate_m2t_scores_batch(sim_matrix_m2t)
    metrics["m2m_score"] = calculate_m2m_scores_batch(sim_matrix_m2m)
    metrics["m2t_top_1"] = calculate_m2t_top_k_batch(sim_matrix_m2t, k=1)
    metrics["m2t_top_3"] = calculate_m2t_top_k_batch(sim_matrix_m2t, k=3)
    
    ### FID + TRANSITION (PER-SAMPLE) ###
    fid_realism_crop_motion_latents_lst = []
    trans_dist_lst = []
    
    for idx, gfeats in enumerate(motions_guofeats):
        # FID+ realism
        realism_latents = calculate_fid_realism(gfeats, tmr_forward, idx, fps)
        fid_realism_crop_motion_latents_lst.append(realism_latents)
        
        # Transition
        joints_local = guofeats_to_joints_local(torch.from_numpy(gfeats))
        joints_local = joints_local - joints_local[:, :, [0]]
        trans_dist = calculate_transition_distance(joints_local)
        if trans_dist is not None:
            trans_dist_lst.append(trans_dist)
    
    # FID aggregation
    if fid_realism_crop_motion_latents_lst:
        fid_realism_latents = np.concatenate(fid_realism_crop_motion_latents_lst)
        mu, cov = calculate_activation_statistics_normalized(fid_realism_latents)
        
        # Compute GT statistics
        gt_motion_latents = compute_latents_batched(motions_guofeats_gt, tmr_forward, batch_size)
        gt_mu, gt_cov = calculate_activation_statistics_normalized(
            gt_motion_latents if isinstance(gt_motion_latents, np.ndarray) else gt_motion_latents.numpy()
        )
        
        metrics["fid+"] = calculate_frechet_distance(
            gt_mu.astype(float), gt_cov.astype(float),
            mu.astype(float), cov.astype(float)
        )
    else:
        metrics["fid+"] = float('inf')
        
    # Transition distance
    metrics["transition"] = torch.concatenate(trans_dist_lst).mean() if trans_dist_lst else 0.0

    nb_sample = sim_matrix_m2m.shape[0] 
    diversity = calculate_diversity(motion_latents, 300 if nb_sample > 300 else 50)
    metrics["diversity"]=diversity 
    
    gt_mu, gt_cov = calculate_activation_statistics(motion_latents_gt.detach().cpu().numpy())
    mu, cov = calculate_activation_statistics(motion_latents.detach().cpu().numpy())
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    metrics["fid"]=fid 
    
    gt_mu_nrom, gt_cov_nrom = calculate_activation_statistics_normalized(motion_latents_gt.detach().cpu().numpy())
    mu_nrom, cov_nrom = calculate_activation_statistics_normalized(motion_latents.detach().cpu().numpy())
    fid_nrom = calculate_frechet_distance(gt_mu_nrom, gt_cov_nrom, mu_nrom, cov_nrom)
    metrics["fid_nrom"]=fid_nrom 
    
    # === MM-DIST (Matching Score) ===
    mm_dist_all = calculate_matching_score(text_latents, motion_latents, sum_all=False)
    metrics["mm_dist"] = float(mm_dist_all.mean())   # average distance per dataset
    
    contrastive = compute_contrastive_metrics_by_protocol(
        sim_matrix_m2t, sent_emb, protocol
    )
    metrics.update(contrastive)
    
    return metrics

def compute_latents_batched(data, tmr_forward, batch_size=256):
    """Compute latents in batches for any data (texts or motions)."""

    n_samples = len(data)
    nsplit = int(np.ceil(n_samples / batch_size))
    
    # Manual split instead of np.array_split to avoid ragged array warning
    data_batches = []
    for i in range(nsplit):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        data_batches.append(data[start:end])
    
    latents = []
    with torch.inference_mode():
        for batch in data_batches:
            batch_latents = tmr_forward(list(batch))
            latents.append(batch_latents)
    
    latents = torch.cat(latents) if isinstance(latents[0], torch.Tensor) else np.concatenate(latents)
    return latents


def compute_sim_matrix_m2t(motions_guofeats, texts, tmr_forward, batch_size=256):
    """Compute motion-to-text similarity matrix."""
    motion_latents = compute_latents_batched(motions_guofeats, tmr_forward, batch_size)
    text_latents = compute_latents_batched(texts, tmr_forward, batch_size)
    sim_matrix = get_sim_matrix(motion_latents, text_latents).numpy()
    return sim_matrix, motion_latents, text_latents


def compute_sim_matrix_m2m(motions_guofeats, motions_guofeats_gt, tmr_forward, batch_size=256):
    """Compute motion-to-motion similarity matrix."""
    motion_latents = compute_latents_batched(motions_guofeats, tmr_forward, batch_size)
    motion_latents_gt = compute_latents_batched(motions_guofeats_gt, tmr_forward, batch_size)
    sim_matrix = get_sim_matrix(motion_latents, motion_latents_gt).numpy()
    return sim_matrix, motion_latents, motion_latents_gt


def compute_contrastive_metrics_by_protocol(sim_matrix_m2t, sent_emb=None, protocol="normal"):
    """Compute contrastive metrics based on protocol."""
    if protocol == "normal":
        metrics = all_contrastive_metrics(
            sims=sim_matrix_m2t.T, emb=None, threshold=None,
            rounding=2, return_cols=False, m2t=True, t2m=True
        )
        return metrics
        
    elif protocol == "threshold":
        metrics = all_contrastive_metrics(
            sims=sim_matrix_m2t.T, emb=sent_emb, threshold=0.85,
            rounding=2, return_cols=False, m2t=True, t2m=True
        )
        return metrics
        
    elif protocol in ["guo", "guo+threshold"]:
        batch_size_guo = 32
        n_samples = sim_matrix_m2t.shape[0]
        
        idx = np.arange(n_samples)
        np.random.seed(0)
        np.random.shuffle(idx)
        
        n_batches = n_samples // batch_size_guo
        idx_batches = [idx[batch_size_guo * i : batch_size_guo * (i + 1)] 
                      for i in range(n_batches)]
        
        all_batch_metrics = []
        threshold_val = 0.85 if "threshold" in protocol else None
        
        for batch_idx in idx_batches:
            batch_sim = sim_matrix_m2t[batch_idx][:, batch_idx]
            batch_emb = sent_emb[batch_idx] if sent_emb is not None and threshold_val else None
            
            batch_metrics = all_contrastive_metrics(
                sims=batch_sim.T, emb=batch_emb, threshold=threshold_val,
                rounding=2, return_cols=False, m2t=True, t2m=True
            )
            all_batch_metrics.append(batch_metrics)
        
        avg_metrics = {}
        for key in all_batch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_batch_metrics])
        
        return avg_metrics
    
    return {}


def evaluate_experiment_enhanced(base_dir, split_path=None, split="val", fps=20.0, max_runs=None, 
                                protocol="normal", batch_size=256, device="cpu",
                                dataset_name="babel-part",
                                base_eval_model_dir="pretrained/eval_model",
                                default_eval_model_dir=None,
                                base_annotations_root="datasets/annotations", y_is_z_axis=False):
    """
    Evaluate experiment with enhanced dataset support.
    
    Args:
        base_dir: Path to experiment directory
        split_path: Path to splits directory
        split: Split name
        fps: Frames per second
        max_runs: Maximum number of runs to evaluate (None = all available)
    """
    print(f"Evaluating experiment: {base_dir}")
    print("="*60)
    
    # Load experiment summary
    summary = load_run_summary(base_dir)
    total_runs_available = summary["num_runs"]
    
    # Determine how many runs to actually evaluate
    if max_runs is None:
        num_runs_to_eval = total_runs_available
    else:
        num_runs_to_eval = min(max_runs, total_runs_available)
    
    print(f"Total runs available: {total_runs_available}")
    print(f"Runs to evaluate: {num_runs_to_eval}")
    
    if num_runs_to_eval == 0:
        raise ValueError("No runs to evaluate")
    # Build model directories for this dataset
    print(f"Dataset name: {dataset_name}")
    model_dirs = get_model_dirs_for_dataset(dataset_name, base_eval_model_dir)
    
    if default_eval_model_dir is None:
        default_eval_model_dir = os.path.join(base_eval_model_dir, "tmr_humanml3d_guoh3dfeats")
        
    # Load experiment data with enhanced support
    print("Loading experiment data...")
    all_data = load_experiment_data_enhanced(
        base_dir, split_path=split_path, split=split, 
        convert_to_guofeats=True, include_gt=True, include_generated=True,
        max_runs=max_runs,
        dataset_name=dataset_name,
        base_annotations_root=base_annotations_root,
        y_is_z_axis=y_is_z_axis
    )
    
    if 'gt' not in all_data:
        raise ValueError("No GT data found in experiment directory")
    
    # Extract GT datasets
    gt_datasets = all_data['gt']
    
    # Print header
    metric_names = [32 * " ", "R1  ", "R3  ", "M2T S", "MM dist", "Diversity", "FID ", "FID+ ", "FID_norm ", "Trans"]
    header = " & ".join(metric_names)
    
    # Evaluate each dataset type
    results = {
        "gt_metrics": {},
        "run_metrics": {dataset_name: [] for dataset_name in gt_datasets.keys()},
        "avg_metrics": {},
        "std_metrics": {}
    }
    
    # 1. CAPTION EVALUATION
    if "caption" in gt_datasets and gt_datasets["caption"][0]:  # Has caption data
        print(f"\n{header}")
        print("CAPTION EVALUATION:")
        
        # gt_caption_texts, gt_caption_motions, _ = gt_datasets["caption"]
        result = gt_datasets["caption"]
        if len(result) == 3:
            gt_caption_texts, gt_caption_motions, _ = result
            sent_emb = None
        else:
            gt_caption_texts, gt_caption_motions, _, sent_emb = result
        
        tmr_forward = load_eval_model_for_dataset("caption", model_dirs, default_eval_model_dir, device)
        
        # GT caption metrics
        gt_caption_metrics = get_gt_metrics_for_dataset(
            gt_caption_texts, gt_caption_motions, tmr_forward, 
            sent_emb=sent_emb, protocol=protocol, batch_size=batch_size
        )
        results["gt_metrics"]["caption"] = gt_caption_metrics
        print_latex("GT Caption", gt_caption_metrics)
        
        # Evaluate each run's caption data
        generated_data = all_data.get('generated', {})
        for run_idx in range(num_runs_to_eval):
            if run_idx in generated_data and "caption" in generated_data[run_idx]:
                # gen_caption_texts, gen_caption_motions, _ = generated_data[run_idx]["caption"]
                if len(result) == 3:
                    gen_caption_texts, gen_caption_motions, _ = generated_data[run_idx]["caption"]
                    sent_emb = None
                else:
                    gen_caption_texts, gen_caption_motions, _, sent_emb = generated_data[run_idx]["caption"]
                    
                if gen_caption_texts:  # Has caption data
                    caption_metrics = calculate_dataset_metrics(
                        gen_caption_texts, gen_caption_motions, tmr_forward,
                        gt_caption_motions,
                        fps, sent_emb=sent_emb, protocol=protocol, batch_size=batch_size
                    )
                    results["run_metrics"]["caption"].append(caption_metrics)
                    print_latex(f"Run {run_idx:02d}", caption_metrics)
        
        # Calculate averages for caption
        if results["run_metrics"]["caption"]:
            results["avg_metrics"]["caption"] = {}
            results["std_metrics"]["caption"] = {}
            for key in results["run_metrics"]["caption"][0].keys():
                values = [m[key] for m in results["run_metrics"]["caption"]]
                results["avg_metrics"]["caption"][key] = np.mean(values)
                results["std_metrics"]["caption"][key] = np.std(values)
            
            print("-" * 60)
            print_latex("Caption AVG", results["avg_metrics"]["caption"])
            print_latex("Caption STD", results["std_metrics"]["caption"])
    
    # 2. ACTION EVALUATION
    if "action" in gt_datasets and gt_datasets["action"][0]:  # Has action data
        print(f"\n{header}")
        print("ACTION EVALUATION:")
        
        # gt_action_texts, gt_action_motions, _ = gt_datasets["action"]
        result = gt_datasets["action"]
        if len(result) == 3:
            gt_action_texts, gt_action_motions, _ = result
            sent_emb = None
        else:
            gt_action_texts, gt_action_motions, _, sent_emb = result
        
        tmr_forward = load_eval_model_for_dataset("action", model_dirs, default_eval_model_dir, device)
        
        # GT action metrics
        gt_action_metrics = get_gt_metrics_for_dataset(
            gt_action_texts, gt_action_motions, tmr_forward,
            sent_emb, protocol=protocol, batch_size=batch_size
        )
        results["gt_metrics"]["action"] = gt_action_metrics
        print_latex("GT Action", gt_action_metrics)

        
        # Evaluate each run's action data
        generated_data = all_data.get('generated', {})
        for run_idx in range(num_runs_to_eval):
            if run_idx in generated_data and "action" in generated_data[run_idx]:
                # gen_action_texts, gen_action_motions, _ = generated_data[run_idx]["action"]
                if len(result) == 3:
                    gen_action_texts, gen_action_motions, _ = generated_data[run_idx]["action"]
                    sent_emb = None
                else:
                    gen_action_texts, gen_action_motions, _, sent_emb = generated_data[run_idx]["action"]
                    
                if gen_action_texts:  # Has action data
                    action_metrics = calculate_dataset_metrics(
                        gen_action_texts, gen_action_motions, tmr_forward,
                        gt_action_motions,  
                        fps, sent_emb, protocol=protocol, batch_size=batch_size
                    )
                    results["run_metrics"]["action"].append(action_metrics)
                    print_latex(f"Run {run_idx:02d}", action_metrics)
        
        # Calculate averages for action
        if results["run_metrics"]["action"]:
            results["avg_metrics"]["action"] = {}
            results["std_metrics"]["action"] = {}
            for key in results["run_metrics"]["action"][0].keys():
                values = [m[key] for m in results["run_metrics"]["action"]]
                results["avg_metrics"]["action"][key] = np.mean(values)
                results["std_metrics"]["action"][key] = np.std(values)
            
            print("-" * 60)
            print_latex("Action AVG", results["avg_metrics"]["action"])
            print_latex("Action STD", results["std_metrics"]["action"])
    
    # 3. PARTS EVALUATION
    print(f"\n{header}")
    print("PARTS EVALUATION:")
    
    parts_gt_metrics = {}
    parts_run_metrics = {part: [] for part in BODY_PARTS}
    
    # Evaluate each body part
    for part in BODY_PARTS:
        if part in gt_datasets and gt_datasets[part][0]:  # Has data for this part
            # gt_part_texts, gt_part_motions, _, _ = gt_datasets[part]
            result = gt_datasets[part]
            if len(result) == 3:
                gt_part_texts, gt_part_motions, _ = result
                sent_emb = None
            else:
                gt_part_texts, gt_part_motions, _, sent_emb = result

            tmr_forward = load_eval_model_for_dataset(part, model_dirs, default_eval_model_dir, device)
            
            # GT part metrics
            gt_part_metrics = get_gt_metrics_for_dataset(
                gt_part_texts, gt_part_motions, tmr_forward, sent_emb,
                protocol=protocol, batch_size=batch_size
            )
            parts_gt_metrics[part] = gt_part_metrics
            print_latex(f"GT {part.title()}", gt_part_metrics)
            
            
            # Evaluate each run's part data
            generated_data = all_data.get('generated', {})
            for run_idx in range(num_runs_to_eval):
                if run_idx in generated_data and part in generated_data[run_idx]:
                    if len(result) == 3:
                        gen_part_texts, gen_part_motions, _ = generated_data[run_idx][part]
                        sent_emb = None
                    else:
                        gen_part_texts, gen_part_motions, _, sent_emb = generated_data[run_idx][part]   
                        
                    if gen_part_texts:  # Has data for this part
                        part_metrics = calculate_dataset_metrics(
                            gen_part_texts, gen_part_motions, tmr_forward,
                            gt_part_motions, 
                            fps, sent_emb, protocol=protocol, batch_size=batch_size
                        )
                        parts_run_metrics[part].append(part_metrics)
    
    # Calculate average across parts (for GT)
    if parts_gt_metrics:
        parts_gt_avg = {}
        for key in list(parts_gt_metrics.values())[0].keys():
            parts_gt_avg[key] = np.mean([metrics[key] for metrics in parts_gt_metrics.values()])
        results["gt_metrics"]["parts_avg"] = parts_gt_avg
        print_latex("GT Parts AVG", parts_gt_avg)
    
    # Calculate average across parts (for each run)
    max_runs = min(num_runs_to_eval, max(len(metrics_list) for metrics_list in parts_run_metrics.values()))
    parts_avg_run_metrics = []
    
    for run_idx in range(max_runs):
        run_avg = {}
        valid_parts = []
        for part in BODY_PARTS:
            if run_idx < len(parts_run_metrics[part]):
                valid_parts.append(parts_run_metrics[part][run_idx])
        
        if valid_parts:
            for key in valid_parts[0].keys():
                run_avg[key] = np.mean([part_metrics[key] for part_metrics in valid_parts])
            parts_avg_run_metrics.append(run_avg)
            print_latex(f"Run {run_idx:02d}", run_avg)
    
    # Calculate final averages and std for parts
    if parts_avg_run_metrics:
        results["avg_metrics"]["parts_avg"] = {}
        results["std_metrics"]["parts_avg"] = {}
        for key in parts_avg_run_metrics[0].keys():
            values = [m[key] for m in parts_avg_run_metrics]
            results["avg_metrics"]["parts_avg"][key] = np.mean(values)
            results["std_metrics"]["parts_avg"][key] = np.std(values)
        
        print("-" * 60)
        print_latex("Parts AVG", results["avg_metrics"]["parts_avg"])
        print_latex("Parts STD", results["std_metrics"]["parts_avg"])
    
    # Store individual parts metrics
    results["gt_metrics"]["parts"] = parts_gt_metrics
    results["run_metrics"]["parts"] = parts_run_metrics
    
    results['num_runs'] = num_runs_to_eval
    results['total_runs_available'] = total_runs_available
    results['summary'] = summary
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated motion results with enhanced dataset support")
    parser.add_argument("experiment_dir", help="Path to experiment directory containing run_summary.json")
    parser.add_argument("--base_splits_root",
                       default="datasets/annotations",
                       help="Root directory for splits")
    parser.add_argument("--base_annotations_root", 
                       default="datasets/annotations",
                       help="Root directory for annotations")
    parser.add_argument("--split", default="test", help="Split name (default: val)")
    parser.add_argument("--fps", type=float, default=20.0, help="Frames per second (default: 20.0)")
    parser.add_argument("--max_runs", type=int, default=20, help="Maximum number of runs to evaluate (default: all available)")
    parser.add_argument("--save_results", default=None, 
                       help="Path to save results JSON file (default: {experiment_dir}/evaluation_results.json)")
    parser.add_argument("--calculate_gt_metrics", action="store_true",
                    help="If set, evaluate GT vs GT (oracle baseline) instead of generated runs.")
    parser.add_argument("--protocol", default="guo", 
                       choices=["normal", "threshold", "guo", "guo+threshold"])
    # Eval is locked to cpu: cuda↔cpu motion-generation parity is not byte-stable
    # across all checkpoints (1-turn R@1 drift up to ~1pt). cpu is the release reference.
    parser.add_argument("--device", default="cpu", choices=["cpu"],
                        help="Eval is cpu-only (release reference). cuda is intentionally disabled.")
    parser.add_argument("--dataset_name", default="amass-part-12", 
                       help="Dataset name (e.g., babel-part, amass-part-12)")
    parser.add_argument("--eval_model_dir", 
                       default="pretrained/eval_model",
                       help="Base directory containing per-body-part eval encoder checkpoints")
    parser.add_argument("--default_eval_model_dir", default=None,
                       help="Default eval encoder directory (fallback)")
    parser.add_argument("--y_up", action="store_true", 
                    help="Set if your data has Y-axis pointing up (default: False, will convert from Z-up)")    
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    
    args.split_path = os.path.join(args.base_splits_root, args.dataset_name)
    # Auto-construct save_results path if not provided
    if args.save_results is None:
        args.save_results = os.path.join(args.experiment_dir, f"evaluation_results_{args.max_runs}_{args.dataset_name}_{args.protocol}.json")
        print(f"Results will be saved to: {args.save_results}")
        
    # Evaluate the experiment
    results = evaluate_experiment_enhanced(
        args.experiment_dir, 
        split_path=args.split_path, 
        split=args.split, 
        fps=args.fps,
        max_runs=args.max_runs,
        protocol=args.protocol,
        batch_size=args.batch_size,
        device=args.device,
        dataset_name=args.dataset_name,
        base_eval_model_dir=args.eval_model_dir,
        base_annotations_root=args.base_annotations_root,
        y_is_z_axis=args.y_up,
    )
    
    # Save results if requested
    if args.save_results:
            save_results_to_csv(results, args.save_results, args)
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Experiment: {os.path.basename(args.experiment_dir)}")
    print(f"Number of runs evaluated: {results['num_runs']}")
    
    # Print summary for each dataset type
    for dataset_type in ["caption", "action", "parts_avg"]:
        if dataset_type in results['avg_metrics'] and results['avg_metrics'][dataset_type]:
            print(f"\n{dataset_type.upper()} - Average metrics across all runs:")
            for metric_name, value in results['avg_metrics'][dataset_type].items():
                if dataset_type in results['std_metrics']:
                    std_val = results['std_metrics'][dataset_type][metric_name]
                    print(f"  {metric_name}: {value:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()
