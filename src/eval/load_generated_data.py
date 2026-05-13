import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# --- motion features ---
from src.tools.guofeats import joints_to_guofeats
from src.tools.guofeats import guofeats_to_joints

# --- torch dataloader bits ---
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

# =========================
# Constants / dataset names
# =========================
BODY_PARTS = ["head", "left_arm", "right_arm", "left_leg", "right_leg", "spine", "trajectory"]
# BODY_PARTS = ["left_arm", "right_arm", "left_leg", "right_leg", "spine"]
BATCH_SIZE = 32  # match reference

# =========================
# Basic IO helpers
# =========================
def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def read_split(path, split):
    # Two layouts supported: tmr_plus-era flat ``<dataset>/splits/val.txt`` and
    # the release layout ``<dataset>/annotations/splits/val.txt``.
    candidates = [
        os.path.join(path, "splits", split + ".txt"),
        os.path.join(path, "annotations", "splits", split + ".txt"),
    ]
    split_file = next((c for c in candidates if os.path.exists(c)), candidates[0])
    id_list = []
    with open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))
    
# =========================
# Availability probe
# =========================
def check_dataset_availability(data_dir, valid_keyids):
    """
    Returns:
        has_captions: bool
        has_actions: bool
    """
    has_captions = True
    has_actions  = True

    print("Checking dataset availability...")
    for keyid in tqdm(valid_keyids, desc="Checking samples"):
        json_path = os.path.join(data_dir, f"{keyid}.json")
        if os.path.exists(json_path):
            annotation = load_json(json_path)
            if annotation.get("caption", "") == "":
                has_captions = False
            if "action_text" not in annotation:
                has_actions = False

    print(f"Dataset availability - Captions: {has_captions}, Actions: {has_actions}")
    return has_captions, has_actions

# =========================
# Directory loader (GT/GEN)
# =========================

# =========================
# Embedding loader helper
# =========================
def load_embeddings_if_requested(load_embeddings, embeddings_folder, embeddings_modelname, base_dir):
    """
    Load embeddings array and index if requested.
    
    Returns:
        embeddings_array: np.ndarray or None
        embeddings_index: dict or None
    """
    if not load_embeddings:
        return None, None
    
    if embeddings_folder is None:
        embeddings_folder = os.path.join(base_dir, "sent_embeddings", "sentence-transformers")
    
    embeddings_path = os.path.join(embeddings_folder, f"{embeddings_modelname}.npy")
    embeddings_index_path = os.path.join(embeddings_folder, f"{embeddings_modelname}_index.json")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(embeddings_index_path):
        # Release bundles don't ship precomputed sentence embeddings — they're
        # only used by the guo+threshold protocol to filter near-duplicate
        # texts, and eval still runs correctly without the filter (just
        # slightly more conservative numbers).
        print(f"  [skip] sentence embeddings not at {embeddings_folder} — guo+threshold filter will be disabled")
        return None, None

    embeddings_array = np.load(embeddings_path)
    embeddings_index = load_json(embeddings_index_path)
    print(f"Loaded embeddings: {embeddings_array.shape} from {embeddings_folder}")

    return embeddings_array, embeddings_index

def load_data_from_directory_enhanced(
    data_dir, split_path=None, split="val",
    skip_unknown=True, convert_to_guofeats=True, y_is_z_axis=False,
    load_embeddings=False,
    embeddings_folder=None,
    embeddings_modelname="all-mpnet-base-v2",
    base_annotations_dir=None
):
    """
    Returns:
        datasets: Dict[str, Tuple[List[str], List[np.ndarray], List[str]]]
                  name -> (texts, motions, keyids)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Load embeddings if requested (and if present on disk — absence is
    # graceful; the guo+threshold filter just gets disabled).
    embeddings_array, embeddings_index = None, None
    if load_embeddings:
        if base_annotations_dir is None:
            raise ValueError("base_annotations_dir must be provided when load_embeddings=True")
        embeddings_array, embeddings_index = load_embeddings_if_requested(
            load_embeddings, embeddings_folder, embeddings_modelname, base_annotations_dir
        )
        
    # Build key universe
    if split_path:
        valid_keyids = read_split(split_path, split)
        print(f"Loaded {len(valid_keyids)} keyids from split: {split}")
    else:
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        valid_keyids = [os.path.splitext(f)[0] for f in npy_files]
        print(f"Found {len(valid_keyids)} keyids in directory")

    if not valid_keyids:
        raise ValueError("No valid keyids found")

    has_captions, has_actions = check_dataset_availability(data_dir, valid_keyids)
    
    if 'total' in split:
        has_captions = True
        has_actions  = True

    datasets: Dict[str, Tuple[List[str], List[np.ndarray], List[str]]] = {}
    if has_captions:
        datasets["caption"] = ([], [], [], []) if load_embeddings else ([], [], [])
    if has_actions:
        datasets["action"] = ([], [], [], []) if load_embeddings else ([], [], [])
    for part in BODY_PARTS:
        datasets[part] = ([], [], [], []) if load_embeddings else ([], [], [])

    print("Loading data...")
    for keyid in tqdm(valid_keyids, desc="Processing samples"):
        joints_path = os.path.join(data_dir, f"{keyid}.npy")
        # Check if the primary path exists, otherwise try the _gt suffix
        if not os.path.exists(joints_path):
            joints_path = os.path.join(data_dir, f"{keyid}_gt.npy")
            if not os.path.exists(joints_path):
                print(f"Warning: Motion file not found for {keyid}")
                continue

        joints = np.load(joints_path, allow_pickle=True)

        # Check if it's a pickled dictionary and extract 'joints' key if it exists
        if isinstance(joints, np.ndarray) and joints.shape == () and joints.dtype == object:
            inner_data = joints.item()
            if isinstance(inner_data, dict) and 'joints' in inner_data:
                joints = inner_data['joints']
                
        if not y_is_z_axis:
            x, y, z = T(joints)
            joints = T(np.stack((x, z, -y), axis=0))
            
        # standardize joint count
        # if joints.shape[1] == 24:
        #     joints = joints[:, :22, :]
        # elif joints.shape[1] != 22:
        #     print(f"Warning: Expected 22 or 24 joints, got {joints.shape[1]} for {keyid}")
        #     continue

        json_path = os.path.join(data_dir, f"{keyid}.json")
        if not os.path.exists(json_path):
            print(f"Warning: No annotation found for {keyid}")
            continue

        annotation = load_json(json_path)

        # Check if motion length matches annotation's sum_length
        sum_length = annotation.get("sum_length")
        if sum_length is not None and len(joints) != sum_length:
            print(f"Warning: Motion length mismatch for {keyid}. "
                f"Motion has {len(joints)} frames, sum_length is {sum_length}. Cropping.")
            joints = joints[:sum_length]

        # convert to guofeats
        if convert_to_guofeats:
            try:
                joints = joints_to_guofeats(joints)
                # Defensive: drop the keyid if conversion produced NaN/Inf in any
                # frame. Rare numerical edge case (~1 in 1600 on the test split)
                # where a specific frame transition trips guofeats; including
                # such a sequence would propagate NaN through the encoder and
                # poison FID covariance computations downstream.
                if not np.all(np.isfinite(joints)):
                    print(f"Warning: NaN/Inf in guofeats for {keyid}, skipping")
                    continue
                if len(joints) > 0:
                    last_frame = joints[-1:, :]
                    joints = np.concatenate([joints, last_frame], axis=0)
            except Exception as e:
                print(f"Error converting {keyid} to guofeats: {e}")
                continue

        # caption
        if has_captions:
            caption = annotation.get("caption", "")
            if caption and caption != "":
                if not skip_unknown or caption.lower() != "unknown":
                    if load_embeddings and embeddings_index is not None and caption not in embeddings_index:
                        continue
                    datasets["caption"][0].append(caption)
                    datasets["caption"][1].append(joints.copy())
                    datasets["caption"][2].append(keyid)
                    if load_embeddings and embeddings_index is not None:
                        emb_idx = embeddings_index[caption]
                        datasets["caption"][3].append(embeddings_array[emb_idx])

        # action segments
        if has_actions:
            action_texts   = annotation.get("action_text", [])
            action_lengths = annotation.get("action_length", [])
            start_frame = 0
            for action_text, action_length in zip(action_texts, action_lengths):
                if not skip_unknown or action_text.lower() != "unknown":
                    if load_embeddings and embeddings_index is not None and action_text not in embeddings_index:
                        continue
                    end_frame = start_frame + action_length
                    if end_frame > len(joints):
                        if end_frame > len(joints)+1:
                            print(f"Warning: Action segment exceeds motion length for {keyid}, action: {action_text}. "
                                f"Expected {end_frame}, available {len(joints)}. Cropping.")
                        end_frame = len(joints)
                    if action_length > 0 and end_frame <= len(joints):
                        cropped = joints[start_frame:end_frame].copy()
                        if len(cropped) > 0:
                            datasets["action"][0].append(action_text)
                            datasets["action"][1].append(cropped)
                            datasets["action"][2].append(f"{keyid}_action_{start_frame}_{end_frame}")
                            if load_embeddings and embeddings_index is not None:
                                emb_idx = embeddings_index[action_text]
                                datasets["action"][3].append(embeddings_array[emb_idx])
                        else:
                            print(f"Warning: Empty action motion for {keyid}, action: {action_text}")
                    else:
                        print(f"Warning: Invalid action length {action_length} for {keyid}, action: {action_text}")
                start_frame += action_length

        # parts
        for part in BODY_PARTS:
            part_texts   = annotation.get(f"{part}_text", [])
            part_lengths = annotation.get(f"{part}_length", [])
            start_frame = 0
            for part_text, part_length in zip(part_texts, part_lengths):
                if not skip_unknown or part_text.lower() != "unknown":
                    if load_embeddings and embeddings_index is not None and part_text not in embeddings_index:
                        continue
                    end_frame = start_frame + part_length
                    if end_frame > len(joints):
                        if end_frame > len(joints)+1:
                            print(f"Warning: Part segment exceeds motion length for {keyid}, part: {part}, text: {part_text}. "
                                    f"Expected {end_frame}, available {len(joints)}. Cropping.")
                        end_frame = len(joints)
                    if part_length > 0 and end_frame <= len(joints):
                        cropped = joints[start_frame:end_frame].copy()
                        if len(cropped) > 0:
                            datasets[part][0].append(part_text)
                            datasets[part][1].append(cropped)
                            datasets[part][2].append(f"{keyid}_{part}_{start_frame}_{end_frame}")
                            if load_embeddings and embeddings_index is not None:
                                emb_idx = embeddings_index[part_text]
                                datasets[part][3].append(embeddings_array[emb_idx])
                        else:
                            print(f"Warning: Empty part motion for {keyid}, part: {part}, text: {part_text}")
                    else:
                        print(f"Warning: Invalid part length {part_length} for {keyid}, part: {part}, text: {part_text}")
                start_frame += part_length

    print("\nDataset Statistics:")
    for dataset_name, data in datasets.items():
        # Handle both (texts, motions, keyids) and (texts, motions, keyids, embeddings)
        texts = data[0]
        print(f"  {dataset_name}: {len(texts)} samples")
    
    return datasets

# =========================
# Run metadata
# =========================
def load_run_summary(base_dir):
    p = os.path.join(base_dir, "run_summary.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"run_summary.json not found, {p}")
    with open(p, "r") as f:
        return json.load(f)

def load_gt_data_enhanced(base_dir, split_path=None, split="val", skip_unknown=True, convert_to_guofeats=True,
                          load_embeddings=False, embeddings_folder=None, embeddings_modelname="all-mpnet-base-v2",
                         base_annotations_dir=None, y_is_z_axis=False):
    gt_dir = os.path.join(base_dir, "gt")
    return load_data_from_directory_enhanced(gt_dir, split_path, split, skip_unknown, convert_to_guofeats,
                                            load_embeddings=load_embeddings,
                                            embeddings_folder=embeddings_folder,
                                            embeddings_modelname=embeddings_modelname,
                                            base_annotations_dir=base_annotations_dir,
                                            y_is_z_axis=y_is_z_axis)

def load_generated_data_enhanced(base_dir, run_idx, split_path=None, split="val", skip_unknown=True, convert_to_guofeats=True,
                                load_embeddings=False, embeddings_folder=None, embeddings_modelname="all-mpnet-base-v2",
                                base_annotations_dir=None, y_is_z_axis=False):
    run_dir = os.path.join(base_dir, f"turn_{run_idx}", "data")
    
    # Check if the data subdirectory exists, otherwise use turn directory directly
    if not os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, f"turn_{run_idx}")
    return load_data_from_directory_enhanced(run_dir, split_path, split, skip_unknown, convert_to_guofeats,
                                            load_embeddings=load_embeddings,
                                            embeddings_folder=embeddings_folder,
                                            embeddings_modelname=embeddings_modelname,
                                            base_annotations_dir=base_annotations_dir,
                                            y_is_z_axis=y_is_z_axis)

def load_all_generated_data_enhanced(base_dir, split_path=None, split="val", skip_unknown=True, convert_to_guofeats=True, max_runs=None,
                                    load_embeddings=False, embeddings_folder=None, embeddings_modelname="all-mpnet-base-v2",
                                    base_annotations_dir=None, y_is_z_axis=False):
    summary = load_run_summary(base_dir)
    total_runs_available = summary["num_runs"]
    num_runs_to_load = total_runs_available if max_runs is None else min(max_runs, total_runs_available)
    print(f"Loading {num_runs_to_load} out of {total_runs_available} available runs")

    all_data = {}
    for run_idx in range(num_runs_to_load):
        try:
            datasets = load_generated_data_enhanced(base_dir, run_idx, split_path, split, skip_unknown, convert_to_guofeats,
                                                   load_embeddings=load_embeddings,
                                                   embeddings_folder=embeddings_folder,
                                                   embeddings_modelname=embeddings_modelname,
                                                   base_annotations_dir=base_annotations_dir,
                                                   y_is_z_axis=y_is_z_axis)
            all_data[run_idx] = datasets
            total_samples = sum(len(data[0]) for data in datasets.values() if data[0])
            print(f"Loaded run {run_idx}: {total_samples} total samples across all datasets")
        except Exception as e:
            print(f"Error loading run {run_idx}: {e}")
    return all_data

def load_experiment_data_enhanced(base_dir, split_path=None, split="val", skip_unknown=True,
                                 convert_to_guofeats=False, include_gt=True, include_generated=True, max_runs=None,
                                 dataset_name="babel-part",
                                 base_annotations_root='datasets/annotations',
                                 fps=20.0,
                                 load_embeddings=True,
                                 embeddings_folder=None,
                                 embeddings_modelname="all-mpnet-base-v2",
                                 y_is_z_axis=False):
    """
    Load all data from an experiment directory with enhanced dataset support.

    Args:
        max_runs: Maximum number of runs to load (None = all available)
        fps: Frames per second for time conversion
    """
    data = {}

    base_annotations_dir = os.path.join(base_annotations_root, dataset_name)

    if include_gt:
        try:
            gt_datasets = load_gt_data_enhanced(base_dir, split_path, split, skip_unknown, convert_to_guofeats,
                                               load_embeddings=load_embeddings,
                                               embeddings_folder=embeddings_folder,
                                               embeddings_modelname=embeddings_modelname,
                                               base_annotations_dir=base_annotations_dir,
                                               y_is_z_axis=y_is_z_axis)
            
            data['gt'] = gt_datasets
            
            total_gt_samples = sum(len(data[0]) for data in gt_datasets.values() if data[0])  # data[0] is texts
            print(f"Loaded GT data: {total_gt_samples} total samples across all datasets")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Error loading GT data: {e}")
    
    if include_generated:
        try:
            all_generated = load_all_generated_data_enhanced(base_dir, split_path, split, skip_unknown, convert_to_guofeats, max_runs,
                                                            load_embeddings=load_embeddings,
                                                            embeddings_folder=embeddings_folder,
                                                            embeddings_modelname=embeddings_modelname,
                                                            base_annotations_dir=base_annotations_dir,
                                                            y_is_z_axis=y_is_z_axis)
            data['generated'] = all_generated
            
            total_runs = len(all_generated)
            print(f"Loaded generated data: {total_runs} runs")
        except Exception as e:
            print(f"Error loading generated data: {e}")
    
    return data

# =========================
# Dataset + Collator
# =========================
class SimpleMotionTextDataset:
    """
    HumanML-like dataset:
      returns (0, 0, text:str, 0, motion[T,D]:Tensor, length:int)
    and provides inv_transform().
    """
    def __init__(self, texts: List[str], motions: List[np.ndarray]):
        self.texts = texts
        self.motions = motions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        motion = self.motions[idx]
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion.astype(np.float32))
        else:
            motion = motion.to(torch.float32)
        length = int(motion.shape[0])
        return 0, 0, self.texts[idx], 0, motion, length

    def inv_transform(self, x):
        # identity; your evaluate_tmr calls this
        return x

def _collate_motion_text(batch):
    """
    Pads variable-length motions to max T in batch.
    Keeps texts as list[str].
    Returns a 6-tuple to match evaluate_tmr's expectations.
    """
    _, _, texts, _, motions, lengths = zip(*batch)
    lengths = [int(l) for l in lengths]
    max_T = max(lengths)
    D = motions[0].shape[-1]
    B = len(motions)

    padded = torch.zeros(B, max_T, D, dtype=torch.float32)
    for i, (m, L) in enumerate(zip(motions, lengths)):
        padded[i, :L] = m[:L]

    lengths_t = torch.tensor(lengths, dtype=torch.long)
    zero = 0
    return zero, zero, list(texts), zero, padded, lengths_t

# =========================
# Sampler builders (shared shuffle)
# =========================
def _make_shared_indices(n_items: int, seed: int) -> List[int]:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n_items, generator=g).tolist()

def _build_sampler_map(datasets: Dict[str, Tuple[List[str], List[np.ndarray], List[str]]], seed: int) -> Dict[str, SubsetRandomSampler]:
    """
    For each dataset name (caption, action, head, ...), create one shared permutation.
    Use the same sampler for GT and all GEN runs -> aligned shuffle.
    """
    sampler_map: Dict[str, SubsetRandomSampler] = {}
    for name, (texts, _motions, _keyids) in datasets.items():
        if texts:
            idx = _make_shared_indices(len(texts), seed)
            sampler_map[name] = SubsetRandomSampler(idx)
    return sampler_map

# =========================
# Loader builders (GT / GEN)
# =========================
def _to_loader(texts, motions, batch_size=BATCH_SIZE, shuffle=False, sampler: Optional[SubsetRandomSampler]=None):
    if not texts:
        return None
    ds = SimpleMotionTextDataset(texts, motions)
    if sampler is not None:
        shuffle = False  # PyTorch: can't set both shuffle & sampler
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=_collate_motion_text,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

def build_loaders_from_datasets(
    datasets: Dict[str, Tuple[List[str], List[np.ndarray], List[str]]],
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler_map: Optional[Dict[str, SubsetRandomSampler]] = None
):
    loaders: Dict[str, Optional[DataLoader]] = {}
    for name, (texts, motions, _keyids) in datasets.items():
        if texts:
            sampler = sampler_map.get(name) if sampler_map else None
            loaders[name] = _to_loader(texts, motions, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        else:
            loaders[name] = None
    return loaders

def load_gt_loaders(
    base_dir, split_path=None, split="val", skip_unknown=True, convert_to_guofeats=True, seed: int = 1234
):
    """
    Returns:
        gt_loaders: Dict[name] -> DataLoader (or None)
        sampler_map: Dict[name] -> SubsetRandomSampler  (reused by GEN loaders to align shuffling)
    """
    gt_datasets = load_gt_data_enhanced(base_dir, split_path, split, skip_unknown, convert_to_guofeats)
    sampler_map = _build_sampler_map(gt_datasets, seed)
    gt_loaders  = build_loaders_from_datasets(gt_datasets, batch_size=BATCH_SIZE, shuffle=False, sampler_map=sampler_map)
    return gt_loaders, sampler_map

def load_run_loaders(
    base_dir, run_idx, split_path=None, split="val", skip_unknown=True, convert_to_guofeats=True, sampler_map: Optional[Dict[str, SubsetRandomSampler]]=None
):
    """
    Returns:
        gen_loaders for a single run, using the provided sampler_map to align with GT.
    """
    gen_datasets = load_generated_data_enhanced(base_dir, run_idx, split_path, split, skip_unknown, convert_to_guofeats)
    return build_loaders_from_datasets(gen_datasets, batch_size=BATCH_SIZE, shuffle=False, sampler_map=sampler_map)

def load_all_run_loaders(
    base_dir, split_path=None, split="val", skip_unknown=True, convert_to_guofeats=True, max_runs=None,
    sampler_map: Optional[Dict[str, SubsetRandomSampler]]=None
):
    """
    Returns:
        all_loaders: Dict[run_idx] -> Dict[name] -> DataLoader or None
        num_runs_loaded: int
        total_runs_available: int
    """
    summary = load_run_summary(base_dir)
    total_runs_available = summary["num_runs"]
    num_runs_to_load = total_runs_available if max_runs is None else min(max_runs, total_runs_available)

    all_loaders: Dict[int, Dict[str, Optional[DataLoader]]] = {}
    for run_idx in range(num_runs_to_load):
        try:
            all_loaders[run_idx] = load_run_loaders(base_dir, run_idx, split_path, split, skip_unknown, convert_to_guofeats, sampler_map=sampler_map)
        except Exception as e:
            print(f"Error building loaders for run {run_idx}: {e}")
    return all_loaders, num_runs_to_load, total_runs_available

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    BASE_DIR   = "/path/to/your/experiment/directory"
    SPLIT_PATH = "/path/to/your/splits"  # contains splits/val.txt, etc.
    SEED       = 1234

    # Build GT loaders + shared permutation (per dataset name)
    gt_loaders, sampler_map = load_gt_loaders(BASE_DIR, SPLIT_PATH, "val", seed=SEED)
    print("GT loaders:", {k: (v is not None) for k, v in gt_loaders.items()})

    # Build GEN loaders for all runs, reusing the same sampler_map -> aligned shuffle
    all_run_loaders, n_loaded, n_total = load_all_run_loaders(
        BASE_DIR, SPLIT_PATH, "val", sampler_map=sampler_map
    )
    print(f"Runs loaded: {n_loaded}/{n_total}")
