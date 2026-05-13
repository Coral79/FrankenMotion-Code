import os
import codecs as cs
import orjson  # loading faster than json
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional

from .collate import collate_text_motion_part


def _resolve_annotations_root(path):
    # Released dataset wraps annotations.json + splits/ inside an annotations/ subdir;
    # internal (stmc) layout has them at the dataset root. Detect either.
    nested = os.path.join(path, "annotations")
    if os.path.isdir(nested) and os.path.exists(os.path.join(nested, "splits")):
        return nested
    return path


def read_split(path, split):
    root = _resolve_annotations_root(path)
    split_file = os.path.join(root, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

def load_annotations_part(path, name="annotations.json"):
    import json
    json_path = os.path.join(_resolve_annotations_root(path), name)
    print('###############Loading From:',json_path)
    with open(json_path, "r", encoding="utf-8") as ff:
        data = json.load(ff)
        
        # Print the count of body parts for verification
        first_key = next(iter(data))
        body_parts = {}
        for ann in data[first_key]["annotations"]:
            bp = ann["bodypart"]
            body_parts[bp] = body_parts.get(bp, 0) + 1
        
        print("Body part counts:", body_parts)
        return data

def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(_resolve_annotations_root(path), name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())

def align(motion, text):
    """
    Aligns the text tensor with the motion tensor along dimension 0 and concatenates them along dimension 1.
    
    Args:
        motion (torch.Tensor): The motion tensor
        text (torch.Tensor): The text embedding tensor
    
    Returns:
        torch.Tensor: Concatenated tensor of aligned motion and text
    """
    # Get the target length from motion's first dimension
    target_length = motion.shape[0]
    current_text_length = text.shape[0]
    
    # Adjust text tensor to match motion's length in dimension 0
    if current_text_length > target_length:
        # If text is longer, truncate it
        text = text[:target_length, :]
    elif current_text_length < target_length:
        # If text is shorter, repeat the last frame to match the length
        last_frame = text[-1:, :]  # Get the last frame, keeping dimensions
        repeat_count = target_length - current_text_length
        repeated_frames = last_frame.repeat(repeat_count, 1)  # Repeat along dim 0
        text = torch.cat([text, repeated_frames], dim=0)
    
    return text

class TextMotionDataset_Part(Dataset):
    def __init__(
        self,
        name: str,
        motion_loader,
        text_encoder,
        split: str = "train",
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
        preload: bool = True,
        tiny: bool = False,
        # only during training
        drop_motion_perc: float = 0.15,
        drop_cond: float = 0.10,
        drop_trans: float = 0.5,
        condition: str = "t2m",
        annotation: str = "annotations.json",
        random_crop_frames: Optional[int] = None,  # for debug df length range
    ):
        path = f"datasets/annotations/{name}"
        self.collate_fn = collate_text_motion_part
        self.split = split
        self.keyids = read_split(path, split)
        if tiny:
            self.keyids = self.keyids[:20]

        self.text_encoder = text_encoder
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations_part(path, annotation)

        # the conditioning information, "t2m", "None"
        self.condition = condition
        self.random_crop_frames = random_crop_frames

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        # if "test" not in split:
        #     self.annotations = self.filter_annotations(self.annotations)
        self.annotations = self.filter_annotations(self.annotations)

        self.is_training = "train" in split
        if drop_motion_perc > 0:
            self.drop_motion_perc = drop_motion_perc
        else:
            self.drop_motion_perc = None
        self.drop_cond = drop_cond
        self.drop_trans = drop_trans

        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        # ADD THESE TWO LINES:
        self._tx_cache = {}  # keyid -> {"x": ..., "mask": ..., "length": int, "text_dict": ...}
        self._tx_uncond = self.text_encoder("unknown")  # Precompute unconditional once
        
        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

        # After preloading is complete
        if preload and motion_loader and not motion_loader.disable:
            for key in motion_loader.motions:
                if isinstance(motion_loader.motions[key], torch.Tensor):
                    motion_loader.motions[key] = motion_loader.motions[key].share_memory_()

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)


    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]
        all_annotations = annotations["annotations"]
        
        # Extract the time range
        start_time = annotations["start"]
        end_time = annotations["end"]
        
        drop_motion_perc = None
        load_transition = False
        empty_annotations = False

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=start_time,
            end=end_time,
            drop_motion_perc=drop_motion_perc,
            load_transition=load_transition,
        )

        # Pass the raw annotations directly to the text encoder
        # Cache or load text embeddings (always unmasked)
        if keyid in self._tx_cache:
            text_emb_dict, text_dict = self._tx_cache[keyid]
        else:
            text_emb_dict, text_dict = self.text_encoder.load_from_annotation(
                annotations=[] if empty_annotations else all_annotations,
                path=annotations["path"],
                start=start_time,
                end=end_time,
            )
            self._tx_cache[keyid] = (text_emb_dict, text_dict)

        # Apply random masking on-the-fly during training (after cache retrieval)
        if self.is_training and self.text_encoder.rand_mask:
            # Make a copy to avoid modifying cache
            text_dict = {k: list(v) for k, v in text_dict.items()}
            text_dict = self.text_encoder.mask_annotations(text_dict)
            
            # Replace masked segments in text embeddings
            text_emb_dict['local']['x'], text_emb_dict['local']['mask'] = \
                self.text_encoder.apply_mask_to_tensor(text_emb_dict['local']['x'], text_dict)
            
        
        # length = motion_x_dict["length"]
        # ADD: Random cropping logic for both motion and text
        if self.random_crop_frames is not None and self.is_training:
            motion = motion_x_dict["x"]
            current_length = motion.shape[0]
            
            if current_length > self.random_crop_frames:
                # Random crop - same indices for both motion and text
                start_idx = torch.randint(0, current_length - self.random_crop_frames + 1, (1,)).item()
                end_idx = start_idx + self.random_crop_frames
                
                # Crop motion
                motion = motion[start_idx:end_idx]
                motion_x_dict["x"] = motion
                motion_x_dict["length"] = self.random_crop_frames
                
                # Crop text embeddings (before alignment)
                text_emb_dict['local']["x"] = text_emb_dict['local']["x"][start_idx:end_idx]
                text_emb_dict['local']["mask"] = text_emb_dict['local']["mask"][start_idx:end_idx]
                text_emb_dict['local']["length"] = self.random_crop_frames
                
                text_dict = self._crop_text_dict(text_dict, start_idx, end_idx)
                
                length = self.random_crop_frames
            else:
                length = motion_x_dict["length"]
        else:
            length = motion_x_dict["length"]        

        motion = motion_x_dict["x"]
        motion_dim = motion.shape[1]

        text = text_emb_dict['local']["x"]
        text = align(motion, text)
        text_dim = text.shape[1]

        x = torch.cat([motion, text], dim=1)
        text_emb_dict['local']["x"] = text

        motion_mask = torch.ones_like(motion)
        text_mask = text_emb_dict['local']["mask"]
        text_mask = align(motion_mask, text_mask)
        stats_mask = torch.cat([motion_mask, text_mask], dim=1)
        text_emb_dict['local']["mask"] = text_mask

        text_uncond_encoded = self._tx_uncond 

        if self.is_training:
            drop_motion_perc = self.drop_motion_perc
            drop_cond = self.drop_cond
            if drop_cond is not None:
                if np.random.binomial(1, drop_cond) == 1:
                    # unconditional generation
                    empty_annotations = True  # Flag to pass empty annotations to text encoder
                    text_emb_dict["x"] = text_uncond_encoded["x"]
                    text_emb_dict["length"] = text_uncond_encoded["length"]



        output = {
            "x": x,
            "motion_dim":motion_dim,
            "text_dim":text_dim,
            "text": text_dict,  # Formatted annotation dictionary 
            "tx": text_emb_dict,  # Text embeddings dictionary with "x" and "length"
            # "tx_uncond": None,
            "tx_uncond": text_uncond_encoded,
            "keyid": keyid,
            "length": length,
            "stats_mask":stats_mask,
            "condition": self.condition,
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]
            # Remove annotations from humanact12 as before
            if "humanact12" in path:
                continue

            # Calculate the overall duration from the top-level start and end times
            duration = val["end"] - val["start"]
            if self.min_seconds <= duration <= self.max_seconds:
                filtered_annotations[key] = val

        return filtered_annotations
    
    def filter_annotations_basic(self, annotations):
        #filter anything less than 2 frames
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]
            # Remove annotations from humanact12 as before
            if "humanact12" in path:
                continue

            # Calculate the overall duration from the top-level start and end times
            duration = val["end"] - val["start"]
            if 2 <= duration*20:
                filtered_annotations[key] = val

        return filtered_annotations

    def _crop_text_dict(self, text_dict, start_idx, end_idx):
        """
        Crop the text_dict annotations to match the cropped motion sequence.
        
        Args:
            text_dict: Dictionary with body part annotations like {'left_arm': [('t pose', 5), ('unknown', 8), ...]}
            start_idx: Start frame index for cropping
            end_idx: End frame index for cropping
        
        Returns:
            Cropped text_dict with adjusted frame counts
        """
        cropped_text_dict = {}
        crop_length = end_idx - start_idx
        
        for body_part, annotations in text_dict.items():
            cropped_annotations = []
            current_frame = 0
            
            for label, frame_count in annotations:
                # Check if this annotation overlaps with crop window
                annotation_start = current_frame
                annotation_end = current_frame + frame_count
                
                # Find overlap with crop window [start_idx, end_idx)
                overlap_start = max(annotation_start, start_idx)
                overlap_end = min(annotation_end, end_idx)
                
                if overlap_start < overlap_end:
                    # There's an overlap, add the overlapping portion
                    overlap_frames = overlap_end - overlap_start
                    cropped_annotations.append((label, overlap_frames))
                
                current_frame += frame_count
                
                # Stop if we've passed the crop window
                if current_frame >= end_idx:
                    break
            
            cropped_text_dict[body_part] = cropped_annotations

        return cropped_text_dict


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
