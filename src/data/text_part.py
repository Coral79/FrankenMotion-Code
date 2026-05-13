import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import orjson
import random

from src.model.backbones.modules.text_encoder import TextToEmb

def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())

def ensure_shared_memory(tensor_dict):
    """Recursively makes all tensors in a dictionary or nested structure shared memory tensors."""
    if isinstance(tensor_dict, dict):
        for key, value in tensor_dict.items():
            tensor_dict[key] = ensure_shared_memory(value)
        return tensor_dict
    elif isinstance(tensor_dict, list):
        return [ensure_shared_memory(item) for item in tensor_dict]
    elif isinstance(tensor_dict, torch.Tensor) and tensor_dict.device.type == "cpu":
        return tensor_dict.share_memory_()
    else:
        return tensor_dict

class TextEmbeddings:
    name = "text_embeddings"
    
    def __init__(
        self,
        base_dir: str,
        dataname: str,
        modelname: str,
        device: str = "cpu",
        preload: bool = True,
        mean_pooling: bool = False,
        disable: bool = False,
        nfeats: Optional[int] = None,
        no_model: bool = False,
        body_part_order: List[str] = ["trajectory", "spine","head", "left_arm", "right_arm", "left_leg", "right_leg"],
        pca: int = 0,
        fps: int = 20,
        umin_s: float = 0.5, 
        umax_s: float = 3.0,
        load_global: bool = False,
        load_global_type: Optional[str] = "action", ##action,caption,none
        rand_mask: bool = False,
        mask_ratio: float = 0.5,
        left_right_strategy: Optional[str] = None
    ):
        assert not mean_pooling
        self.mean_pooling = mean_pooling    
        self.fps = fps
        self.base_dir = base_dir
        self.texts: Dict[str, torch.Tensor] = {}
        self.disable = disable
        self.nfeats = nfeats
        self.pca = pca
        self.device = device
        self.no_model = no_model
        self.load_global = load_global
        self.load_global_type = load_global_type
        self.left_right_strategy = left_right_strategy
        
        self.modelname = modelname
        # Released layout: datasets/annotations/{dataname}/text_embeddings/{modelname}/
        # Internal (stmc) layout: datasets/annotations/{dataname}/text_embeddings/{modelname}_embeddings/
        text_emb_root = os.path.join("datasets/annotations", dataname, "text_embeddings")
        released = os.path.join(text_emb_root, self.modelname)
        internal = os.path.join(text_emb_root, f"{self.modelname}_embeddings")
        self.embeddings_folder = released if os.path.isdir(released) else internal
        self.cache = {}
        self.dataname = dataname
        self.rand_mask = rand_mask
        self.mask_ratio = mask_ratio

        # Fixed ordering of body parts (and order of their embedding chunks)
        self.body_part_order = body_part_order

        # Determine features per part and total features in the concatenated vector
        self.features_per_part = nfeats if nfeats else 512
        self.total_features = self.features_per_part * len(self.body_part_order)
        
        # Calculate minimum and maximum duration in frames
        self.umin = max(1, int(self.fps * umin_s))
        self.umax = int(self.fps * umax_s)

        self.pca_components: Optional[PCA] = None
        if preload and not disable:
            if 'clip' in self.modelname:
                self.embeddings = torch.from_numpy(np.load(os.path.join(self.embeddings_folder, "clip.npy"))).to(dtype=torch.float, device=self.device)
                self.embeddings_index = load_json(os.path.join(self.embeddings_folder, "clip_index.json"))
                self.embeddings_slice = np.load(os.path.join(self.embeddings_folder, "clip_slice.npy"))
            else:
                self.embeddings = torch.from_numpy(np.load(os.path.join(self.embeddings_folder, f"{self.modelname}-{nfeats}.npy"))).to(dtype=torch.float, device=self.device)
                self.embeddings_index = load_json(os.path.join(self.embeddings_folder, f"{self.modelname}_index.json"))
                self.embeddings_slice = np.load(os.path.join(self.embeddings_folder, f"{self.modelname}-{nfeats}_slice.npy"))
        else:
            self.embeddings_index = {}

        # In TextEmbeddings.__init__ after loading embeddings
        if preload and not disable and self.device == "cpu":
            self.embeddings = self.embeddings.share_memory_()

        if self.pca > 0:
            try:
                # Create PCA with target dimensionality per body part
                pca_model = PCA(n_components=self.pca)
                pca_model.fit(self.embeddings)
                print(f"Calculating PCA with target dim per part = {self.pca}")
                self.pca_components = pca_model
            except Exception as e:
                raise RuntimeError(f"Error initializing PCA: {e}")
        

    def _apply_left_right_masking(
        self,
        ann_list_dict_all: Dict[str, List[Tuple[str, int]]],
        strategy: str,
        mask_token: str
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Apply left/right masking strategy."""
        
        if strategy == "simple":
            return self._simple_left_right_masking(ann_list_dict_all, mask_token)
        elif strategy == "temporal":
            return self._temporal_left_right_masking(ann_list_dict_all, mask_token)
        else:
            raise ValueError(f"Unknown left_right_strategy: {strategy}")

    def _simple_left_right_masking(
        self,
        ann_list_dict_all: Dict[str, List[Tuple[str, int]]],
        mask_token: str
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Simple strategy: For each body part pair, randomly choose left or right
        for the entire sequence.
        """
        result = ann_list_dict_all.copy()
        
        # Define body part pairs
        pairs = [
            ("left_leg", "right_leg"),
            ("left_arm", "right_arm")
        ]
        
        for left_part, right_part in pairs:
            if left_part in result and right_part in result:
                # Flip coin for this specific pair: True = keep left, False = keep right
                keep_left = random.random() < 0.5
                
                if keep_left:
                    # Mask all right part runs
                    result[right_part] = [(mask_token, count) for _, count in result[right_part]]
                else:
                    # Mask all left part runs
                    result[left_part] = [(mask_token, count) for _, count in result[left_part]]
        
        return result

    def _temporal_left_right_masking(
        self,
        ann_list_dict_all: Dict[str, List[Tuple[str, int]]],
        mask_token: str
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Temporal strategy: Resolve overlaps chronologically by flipping coins
        for each overlapping time period.
        """
        result = {k: v.copy() for k, v in ann_list_dict_all.items()}
        
        # Define body part pairs
        pairs = [
            ("left_leg", "right_leg"),
            ("left_arm", "right_arm")
        ]
        
        for left_part, right_part in pairs:
            if left_part in result and right_part in result:
                result[left_part], result[right_part] = self._resolve_temporal_overlap(
                    result[left_part], result[right_part], mask_token
                )
        
        return result

    def _resolve_temporal_overlap(
        self,
        left_runs: List[Tuple[str, int]],
        right_runs: List[Tuple[str, int]],
        mask_token: str
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Resolve temporal overlaps between left and right runs.
        For each overlap conflict, flip a coin and mask ALL overlapping segments 
        from the losing side. Segments cannot be broken - entire segments are kept or masked.
        """
        # Convert runs to timeline format: [(start_frame, end_frame, label, original_index)]
        def runs_to_timeline(runs):
            timeline = []
            current_frame = 0
            for idx, (label, count) in enumerate(runs):
                timeline.append((current_frame, current_frame + count - 1, label, idx))
                current_frame += count
            return timeline
        
        left_timeline = runs_to_timeline(left_runs)
        right_timeline = runs_to_timeline(right_runs)
        
        # Track which segment indices to mask for each side
        masked_left_indices = set()
        masked_right_indices = set()
        
        # Get total timeline length
        max_time = 0
        if left_timeline:
            max_time = max(max_time, left_timeline[-1][1])
        if right_timeline:
            max_time = max(max_time, right_timeline[-1][1])
        
        current_time = 0
        while current_time <= max_time:
            # Find active segments at current_time that haven't been masked
            active_left = [seg for seg in left_timeline 
                        if seg[0] <= current_time <= seg[1] and seg[3] not in masked_left_indices]
            active_right = [seg for seg in right_timeline 
                        if seg[0] <= current_time <= seg[1] and seg[3] not in masked_right_indices]
            
            if active_left and active_right:
                # Overlap detected! Find ALL segments involved in this conflict
                
                # Get the full time range of the conflict by finding all overlapping segments
                conflict_left = set()
                conflict_right = set()
                
                # Start with currently active segments
                for seg in active_left:
                    conflict_left.add(seg[3])
                for seg in active_right:
                    conflict_right.add(seg[3])
                
                # Expand to find all segments that overlap with any of these
                changed = True
                while changed:
                    changed = False
                    conflict_segments = []
                    
                    # Collect all segments currently in conflict
                    for seg in left_timeline:
                        if seg[3] in conflict_left:
                            conflict_segments.append(seg)
                    for seg in right_timeline:
                        if seg[3] in conflict_right:
                            conflict_segments.append(seg)
                    
                    if not conflict_segments:
                        break
                        
                    # Find the full time range of current conflict
                    min_start = min(seg[0] for seg in conflict_segments)
                    max_end = max(seg[1] for seg in conflict_segments)
                    
                    # Find any additional segments that overlap with this range
                    for seg in left_timeline:
                        if (seg[0] <= max_end and seg[1] >= min_start and 
                            seg[3] not in masked_left_indices and 
                            seg[3] not in conflict_left):
                            conflict_left.add(seg[3])
                            changed = True
                    
                    for seg in right_timeline:
                        if (seg[0] <= max_end and seg[1] >= min_start and 
                            seg[3] not in masked_right_indices and 
                            seg[3] not in conflict_right):
                            conflict_right.add(seg[3])
                            changed = True
                
                # Now flip coin for entire conflict
                keep_left = random.random() < 0.5
                
                if keep_left:
                    # Mask all conflicting right segments
                    masked_right_indices.update(conflict_right)
                else:
                    # Mask all conflicting left segments  
                    masked_left_indices.update(conflict_left)
                
                # Find the end of this conflict period to jump past it
                conflict_segments = []
                for seg in left_timeline:
                    if seg[3] in conflict_left:
                        conflict_segments.append(seg)
                for seg in right_timeline:
                    if seg[3] in conflict_right:
                        conflict_segments.append(seg)
                
                if conflict_segments:
                    current_time = max(seg[1] for seg in conflict_segments) + 1
                else:
                    current_time += 1
            else:
                current_time += 1
        
        # Convert back to run format
        new_left_runs = []
        for idx, (label, count) in enumerate(left_runs):
            if idx in masked_left_indices:
                new_left_runs.append((mask_token, count))
            else:
                new_left_runs.append((label, count))
        
        new_right_runs = []
        for idx, (label, count) in enumerate(right_runs):
            if idx in masked_right_indices:
                new_right_runs.append((mask_token, count))
            else:
                new_right_runs.append((label, count))
        
        return new_left_runs, new_right_runs
    
    def mask_annotations(
        self,
        ann_list_dict_all: Dict[str, List[Tuple[str, int]]],
        mask_token: str = "unknown",
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Randomly mask runs of labels using a Bernoulli process.
        Each run has an independent probability `mask_ratio` of being replaced by mask_token.
        
        Args:
            ann_list_dict_all: dict mapping body part -> list of (label, num_frames)
            mask_token: replacement label (default: "unknown")
            left_right_strategy: strategy for left/right masking
                - None: no left/right masking (default behavior)
                - "simple": one random choice per body part pair for entire sequence
                - "temporal": resolve overlaps chronologically

        Returns:
            A new dict with the same structure but some runs replaced by mask_token.
        """
        # Step 1: Apply left/right masking if enabled
        if self.left_right_strategy is not None:
            ann_list_dict_all = self._apply_left_right_masking(
                ann_list_dict_all, self.left_right_strategy, mask_token
            )
        
        # Step 2: Sample dynamic masking ratio for this sample (averages to self.mask_ratio)
        alpha = self.mask_ratio * 5
        beta = (1 - self.mask_ratio) * 5
        dynamic_ratio = np.random.beta(alpha, beta)

        # Step 3: Apply masking with the sampled ratio
        masked_dict = {}
        for bodypart, runs in ann_list_dict_all.items():
            new_runs = []
            for label, count in runs:
                if random.random() < dynamic_ratio:
                    new_runs.append((mask_token, count))
                else:
                    new_runs.append((label, count))
            masked_dict[bodypart] = new_runs

        return masked_dict


    def load_from_annotation(
        self, 
        annotations: Optional[List[Dict[str, Any]]] = None, 
        path: Optional[str] = None, 
        start: Optional[float] = None, 
        end: Optional[float] = None, 
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Process raw annotations and text embeddings.
        Also creates a mask based on the annotations.
        Returns a dictionary x_dict with:
        - 'x': the text embeddings (after optional PCA and zeroing masked parts)
        - 'length': number of frames
        - 'mask': a tensor with the same shape as x, where each frame's part gets a 0 if
                    the corresponding label is "unkown" and 1 otherwise.
        And returns a text dictionary (with string descriptions per body part).
        """
        # Determine the total number of frames
        duration_frames = int(self.fps * (end - start)) if (start is not None and end is not None) else self.umin
        
        # Process annotations into both a text dictionary (for display) and an annotation list (for mask)
        ann_list_dict_all = self._format_annotations(annotations, start, end, self.fps) if annotations else ({}, {})
        # Filter the text_dict to include only the expected body parts
        # if self.rand_mask:
        #     ann_list_dict_part = self.mask_annotations(ann_list_dict_all)
        # else:
        #     ann_list_dict_part = ann_list_dict_all
            
        # ann_list_dict = {part: ann_list for part, ann_list in ann_list_dict_part.items() if part in self.body_part_order}
        
        ann_list_dict = {part: ann_list for part, ann_list in ann_list_dict_all.items() if part in self.body_part_order}
        
        # Generate the mask tensor (one-dimensional per part, then expanded to the part's full feature size)
        mask = self._generate_mask(ann_list_dict, duration_frames)
        
        # Load or create the text embedding tensor
        if self.disable:
            # When disabled simply return the file path and a duration
            dummy = {"x": path, "length": duration_frames}
            return dummy, ann_list_dict   

        # MODIFIED SECTION: Generate text data from clip embeddings using annotations
        # Create an empty tensor for text_data
        text_data = torch.zeros((duration_frames, self.total_features), dtype=torch.float, device=self.device)

        # Fill text_data for each body part based on annotations
        for part_idx, part in enumerate(self.body_part_order):
            part_start = part_idx * self.features_per_part
            part_end = (part_idx + 1) * self.features_per_part
            
            # If this body part has annotations
            if part in ann_list_dict:
                ann_list = ann_list_dict[part]
                current_frame = 0  # Reset frame counter for each body part
                last_embedding_tensor = None  # Store the last embedding for padding
                
                # Process each annotation for this body part
                for label, nframes in ann_list:
                    # Get the embedding for this label
                    embedding_dict = self.get_embedding(label)
                    
                    embedding = embedding_dict["x"]
                    # Convert numpy arrays to torch tensors if needed
                    if isinstance(embedding, np.ndarray):
                        embedding_tensor = torch.from_numpy(embedding).float()
                    else:
                        embedding_tensor = embedding.float()
                    
                    # Keep track of the last embedding
                    last_embedding_tensor = embedding_tensor
                    
                    # Calculate end frame without exceeding duration
                    end_frame = min(current_frame + nframes, duration_frames)
                    frames_to_fill = end_frame - current_frame
                    
                    # Fill all frames at once without a loop - much more efficient
                    text_data[current_frame:end_frame, part_start:part_end] = embedding_tensor.expand(frames_to_fill, -1)
                    
                    current_frame = end_frame
                    if current_frame >= duration_frames:
                        break
                
                # EXPLICIT CHECK: After processing all annotations, see if we need padding
                # If we haven't filled the entire duration, pad with the last embedding
                if current_frame < duration_frames and last_embedding_tensor is not None:
                    remaining_frames = duration_frames - current_frame
                    text_data[current_frame:duration_frames, part_start:part_end] = last_embedding_tensor.unsqueeze(0).expand(remaining_frames, -1)
            else:
                # If no annotations for this part, raise an error
                raise ValueError(f"No annotations found for body part '{part}'. Each body part must have annotations.")
        
        # Convert to device
        text_data = text_data.to(self.device)

        # Sanity check: for each body part, if the mask is 0, the embedding for that part should be unknown.
        # self._sanity_check(text_data, mask)

        # Apply PCA if specified
        if self.pca > 0 and self.pca_components is not None:
            text_data,reduced_mask = self._apply_pca(text_data, mask)
        else:
            reduced_mask = mask
        
        # Return the embeddings along with their lengths and the mask.
        x_dict ={} 
        
        if self.load_global:
            # action_list_dict = {part: ann_list for part, ann_list in ann_list_dict_all.items() if part == "action"}
            # ann_list = action_list_dict['action']
            if self.load_global_type == "action":
                action_list_dict = {part: ann_list for part, ann_list in ann_list_dict_all.items() if part == "action"}
                ann_list = action_list_dict['action']
            else:
                action_list_dict = {part: ann_list for part, ann_list in ann_list_dict_all.items() if part == 'sequence_caption'}
                ann_list = action_list_dict['sequence_caption']
            for label, nframes in ann_list:
                # Precomputed in advance
                if label in self:
                    x_dict = self.get_embedding(label)
                # Already computed during the session
                elif label in self.cache:
                    x_dict = self.cache[label]
                # Load the text model (if not already loaded) + compute on the fly
                else:
                    # if self.no_model:
                    #     raise ValueError("This text is not precomputed")
                    print(label)
                    model = self.get_model()
                    x_dict = model(label)
                    self.cache[label] = x_dict
                # global_dict = self.get_embedding(label) 
                x_dict['text'] = label
                global_dict = x_dict  
                x_dict = global_dict
        
        x_dict['local']={"x": text_data, "length": text_data.shape[0], "mask": reduced_mask}

        return x_dict, ann_list_dict

    def __contains__(self, text):
        return text in self.embeddings_index

    def get_model(self):
        model = getattr(self, "model", None)
        if model is None:
            model = self.load_model()
        return model

    def load_model(self):
            self.model = TextToEmb(
                self.modelname, mean_pooling=self.mean_pooling, device=self.device
            )
            return self.model

    def _organize_annotations(self, annotations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group annotations by body part and sort them by start time.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ann in annotations:
            bodypart = ann.get("bodypart")
            if bodypart is None:
                continue
            grouped.setdefault(bodypart, []).append(ann)
        
        for part in grouped:
            grouped[part].sort(key=lambda x: x.get("start", 0))
        return grouped

    def _format_annotations(
        self, 
        annotations: List[Dict[str, Any]], 
        start_time: Optional[float], 
        end_time: Optional[float], 
        fps: int
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Format annotations into a dictionary that maps each body part to a list of tuples (label, num_frames).
        This method first creates frame-by-frame annotations and then compresses them into runs of consecutive
        frames with the same label.
        """
        # Organize annotations by body part
        organized = self._organize_annotations(annotations)
        ann_list_dict: Dict[str, List[Tuple[str, int]]] = {}
        
        # Default to umin frames if start/end not provided
        if start_time is None or end_time is None:
            total_frames = self.umin
        else:
            total_frames = int((end_time - start_time) * fps)
        
        # Generate frame times (in seconds)
        frame_times = [start_time + (i / fps) for i in range(total_frames)] if start_time is not None else []
        
        # Process annotations for each body part
        for bodypart, anns in organized.items():
            # Skip empty annotations
            if not anns:
                continue
            
            # Get text annotation for each frame
            frame_texts = []
            for time in frame_times:
                # Default text is "unknown"
                text = "unknown"
                # Find applicable annotation
                for ann in anns:
                    ann_start = ann.get("start", 0)
                    ann_end = ann.get("end", 0)
                    if ann_start <= time < ann_end:
                        text = ann.get("text", "unknown")
                        break
                frame_texts.append(text)
            
            # Compress consecutive frames with the same label into (label, count) tuples
            if frame_texts:
                compressed_list = []
                current_label = frame_texts[0]
                current_count = 1
                
                # Process from the second frame onwards
                for i in range(1, len(frame_texts)):
                    if frame_texts[i] == current_label:
                        # Same label, increment count
                        current_count += 1
                    else:
                        # Different label, add the current run and start a new one
                        compressed_list.append((current_label, current_count))
                        current_label = frame_texts[i]
                        current_count = 1
                
                # Add the last run
                compressed_list.append((current_label, current_count))
                
                # Store in the result dictionary
                ann_list_dict[bodypart] = compressed_list
        
        return ann_list_dict

    def _generate_mask(self, 
                       ann_list_dict: Dict[str, List[Tuple[str, int]]], 
                       total_frames: int
                      ) -> torch.Tensor:
        """
        For each body part, generate a 1D mask vector of length total_frames.
        This mask is 1 for frames where the annotation label is not "unkown" and 0 where it is.
        If the annotations for a part do not cover all total_frames, the remaining frames are set to 0.
        The final mask tensor will have shape (total_frames, total_features),
        with each body part’s mask repeated for its corresponding feature chunk.
        """
        mask_parts = []
        # Process each body part in order
        for part in self.body_part_order:
            part_mask = torch.zeros(total_frames, dtype=torch.float32)
            pointer = 0
            # Get the list of (label, num_frames) for this body part if available
            ann_list = ann_list_dict.get(part, [])
            for label, nframes in ann_list:
                # Ensure not to exceed total_frames
                end_frame = min(pointer + nframes, total_frames)
                # If the label is not "unknown", mark as 1; otherwise keep 0.
                value = 0 if label == "unknown" else 1
                part_mask[pointer:end_frame] = value
                pointer = end_frame
            # If the annotations did not fill the total_frames, the rest stays 0.
            # Repeat the 1D mask for the feature dimension of this part.
            part_mask_expanded = part_mask.unsqueeze(1).repeat(1, self.features_per_part)
            mask_parts.append(part_mask_expanded)
        # Concatenate the masks from all body parts along the feature dimension
        mask_full = torch.cat(mask_parts, dim=1).to(self.device)
        return mask_full

    def _apply_pca(self, text_data: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply PCA dimension reduction for each body part separately without using a loop.
        For masked parts (mask=0), replace with zeros after PCA transformation.
        Expects text_data of shape (frames, total_features) and mask of the same shape.
        Returns the reduced data and a corresponding reduced mask.
        """
        frames = text_data.shape[0]
        original_features_per_part = self.features_per_part
        reduced_dim_per_part = self.pca
        reduced_features = reduced_dim_per_part * len(self.body_part_order)
        
        # Create the output tensor for reduced data
        reduced_data = torch.zeros((frames, reduced_features), dtype=torch.float32, device=self.device)
        
        # Create the output tensor for reduced mask
        reduced_mask = torch.zeros((frames, reduced_features), dtype=torch.float32, device=self.device)
        
        # Convert the entire tensor to numpy for PCA processing
        text_data_np = text_data.cpu().numpy()
        
        # Process each body part
        for i, part in enumerate(self.body_part_order):
            # Extract features for current body part
            col_start = i * original_features_per_part
            col_end = (i + 1) * original_features_per_part
            
            # Extract the mask for this part (just need first column since mask is same for all features in a part)
            part_mask = mask[:, col_start].cpu().numpy().reshape(-1, 1)  # shape: (frames, 1)
            
            # Apply PCA to all frames at once for this body part
            part_features = text_data_np[:, col_start:col_end]
            transformed = self.pca_components.transform(part_features)
            
            # Take only the needed dimensions
            transformed = transformed[:, :reduced_dim_per_part]
            
            # Apply mask: zero out the rows where mask is 0
            # Broadcasting part_mask to match transformed shape
            masked_transformed = transformed * part_mask
            
            # Store the result in the appropriate slice of reduced_data
            reduced_slice = slice(i * reduced_dim_per_part, (i + 1) * reduced_dim_per_part)
            reduced_data[:, reduced_slice] = torch.from_numpy(masked_transformed)
            
            # Create the corresponding reduced mask (expand part_mask to match reduced dimensions)
            # We repeat the mask values across all reduced dimensions for this body part
            reduced_mask[:, reduced_slice] = mask[:, col_start:col_start+1].repeat(1, reduced_dim_per_part)
        
        return reduced_data, reduced_mask

    def apply_mask_to_tensor(self, text_data: torch.Tensor, text_dict_masked: Dict[str, List[Tuple[str, int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to a pre-built text_data tensor by zeroing out masked segments.
        Also generates the corresponding mask tensor.
        
        Args:
            text_data: Unmasked text embedding tensor (frames, total_features)
            text_dict_masked: Masked annotation dict with 'unknown' labels where masked
        
        Returns:
            Tuple of (masked_text_data, mask_tensor)
        """
        frames = text_data.shape[0]
        masked_text_data = text_data.clone()
        
        # Determine features per part from the actual text_data shape
        actual_features_per_part = text_data.shape[1] // len(self.body_part_order)
        
        # Generate mask and zero out masked segments
        mask_parts = []
        for part_idx, part in enumerate(self.body_part_order):
            part_start = part_idx * actual_features_per_part
            part_end = (part_idx + 1) * actual_features_per_part
            
            part_mask = torch.zeros(frames, dtype=torch.float32)
            
            if part in text_dict_masked:
                current_frame = 0
                for label, nframes in text_dict_masked[part]:
                    end_frame = min(current_frame + nframes, frames)
                    
                    if label == "unknown":
                        # Just zero out masked segments
                        masked_text_data[current_frame:end_frame, part_start:part_end] = 0
                        part_mask[current_frame:end_frame] = 0
                    else:
                        part_mask[current_frame:end_frame] = 1
                    
                    current_frame = end_frame
                    if current_frame >= frames:
                        break
            
            part_mask_expanded = part_mask.unsqueeze(1).repeat(1, actual_features_per_part)
            mask_parts.append(part_mask_expanded)
        
        mask_full = torch.cat(mask_parts, dim=1).to(self.device)
        return masked_text_data, mask_full

    def _sanity_check(self, text_data: torch.Tensor, mask: torch.Tensor) -> None:
        """
        For each body part and each frame, verify the following:
        - If mask is 0, the embedding should match the 'unknown' embedding for that body part
        - If mask is 1, the embedding should contain non-zero values (regular embedding)
        
        This check is performed separately for each body part.
        """
        frames = text_data.shape[0]
        features_per_part = self.features_per_part
        
        # Get the 'unknown' embedding once to use for comparison
        unknown_embedding = torch.from_numpy(self.get_embedding("unknown")["x"])
        
        for i, part in enumerate(self.body_part_order):
            col_start = i * features_per_part
            col_end = (i + 1) * features_per_part
            part_data = text_data[:, col_start:col_end]
            part_mask = mask[:, col_start]
            
            
            for j in range(frames):
                if part_mask[j].item() == 0:
                    # For masked parts (mask=0), the embedding should match the 'unknown' embedding
                    assert torch.allclose(part_data[j].float(), unknown_embedding.float(), atol=1e-2), \
                        f"Sanity check failed at frame {j} for part {part}: " \
                        f"expected 'unknown' embedding but got different values."
                else:
                    # For unmasked parts (mask=1), the embedding should be different from the 'unknown' embedding
                    # This is a more flexible check than requiring non-zero values
                    assert not torch.allclose(part_data[j].float(), unknown_embedding.float(), atol=1e-2), \
                        f"Sanity check failed at frame {j} for part {part}: " \
                        f"expected non-'unknown' embedding but got 'unknown' values."

    def get_embedding(self, text):
        # Check if the text exists in the precomputed embeddings
        if text in self.embeddings_index:
            index = self.embeddings_index[text]
            begin, end = self.embeddings_slice[index]
            embedding = self.embeddings[begin:end]
        else:
            # Fall back to 'unknown' if text not found
            print(f"Warning: Text '{text}' not found in precomputed embeddings, using 'unknown' instead")
            if 'unknown' in self.embeddings_index:
                index = self.embeddings_index['unknown']
                begin, end = self.embeddings_slice[index]
                embedding = self.embeddings[begin:end]
            else:
                raise KeyError(f"Neither '{text}' nor 'unknown' found in precomputed embeddings")
        
        x_dict = {"x": embedding, "length": len(embedding)}
        return x_dict

    def __call__(self, texts):
        if self.disable:
            return texts

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        x_dict_lst = []
        # one at a time here
        for text in texts:
            # Precomputed in advance
            if text in self:
                x_dict = self.get_embedding(text)
            # Already computed during the session
            elif text in self.cache:
                x_dict = self.cache[text]
            # Load the text model (if not already loaded) + compute on the fly
            else:
                if self.no_model:
                    raise ValueError("This text is not precomputed")
                model = self.get_model()
                x_dict = model(text)
                self.cache[text] = x_dict
            x_dict_lst.append(x_dict)

        if squeeze:
            return x_dict_lst[0]
        return x_dict_lst


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    return load_json(json_path)


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))


def save_text_embeddings(path, modelname, device="cuda"):
    model = TextToEmb(modelname, device=device)
    annotations = load_annotations(path)

    path = os.path.join(path, TextEmbeddings.name)
    ptpath = os.path.join(path, f"{modelname}.npy")
    slicepath = os.path.join(path, f"{modelname}_slice.npy")
    jsonpath = os.path.join(path, f"{modelname}_index.json")

    # modelname can have folders
    path = os.path.split(ptpath)[0]
    os.makedirs(path, exist_ok=True)

    # fetch all the texts
    all_texts = [""]
    for dico in annotations.values():
        for lst in dico["annotations"]:
            all_texts.append(lst["text"])

    # remove duplicates
    all_texts = list(set(all_texts))

    # batch of N/10
    nb_tokens = []
    all_texts_batched = np.array_split(all_texts, 100)

    nb_tokens_so_far = 0
    big_tensor = []
    index = []
    for all_texts_batch in tqdm(all_texts_batched):
        x_dict = model(list(all_texts_batch))

        if isinstance(x_dict, torch.Tensor):
            # sentence embeddings
            assert len(x_dict.shape) == 2
            tensor = x_dict[:, None]
            nb_tokens = torch.tensor([1 for x in range(len(tensor))])
        else:
            tensor = x_dict["x"]
            nb_tokens = x_dict["length"]

        # remove padding
        tensor_no_padding = [x[:n].cpu() for x, n in zip(tensor, nb_tokens)]
        tensor_concat = torch.cat(tensor_no_padding)

        big_tensor.append(tensor_concat)
        # save where it is
        ends = torch.cumsum(nb_tokens, 0)
        begins = torch.cat((0 * ends[[0]], ends[:-1]))

        # offset
        ends += nb_tokens_so_far
        begins += nb_tokens_so_far
        nb_tokens_so_far += len(tensor_concat)

        index.append(torch.stack((begins, ends)).T)

    big_tensor = torch.cat(big_tensor).cpu().numpy()
    index = torch.cat(index).cpu().numpy()

    np.save(ptpath, big_tensor)
    np.save(slicepath, index)
    print(f"{ptpath} written")
    print(f"{slicepath} written")

    # correspondance
    dico = {txt: i for i, txt in enumerate(all_texts)}
    write_json(dico, jsonpath)
    print(f"{jsonpath} written")

    #######this function collect text data from the path######
    # def __call__(
    #     self, 
    #     annotations: Optional[List[Dict[str, Any]]] = None, 
    #     path: Optional[str] = None, 
    #     start: Optional[float] = None, 
    #     end: Optional[float] = None, 
    #     **kwargs
    # ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    #     """

    #     Process raw annotations and text embeddings.
    #     Also creates a mask based on the annotations.
    #     Returns a dictionary x_dict with:
    #       - 'x': the text embeddings (after optional PCA and zeroing masked parts)
    #       - 'length': number of frames
    #       - 'mask': a tensor with the same shape as x, where each frame’s part gets a 0 if
    #                 the corresponding label is "unkown" and 1 otherwise.
    #     And returns a text dictionary (with string descriptions per body part).
    #     """
    #     # Determine the total number of frames
    #     duration_frames = int(self.fps * (end - start)) if (start is not None and end is not None) else self.umin
        
    #     # Process annotations into both a text dictionary (for display) and an annotation list (for mask)
    #     ann_list_dict = self._format_annotations(annotations, start, end, self.fps) if annotations else ({}, {})
    #     # Filter the text_dict to include only the expected body parts
    #     ann_list_dict = {part: ann_list for part, ann_list in ann_list_dict.items() if part in self.body_part_order}
        
    #     # Generate the mask tensor (one-dimensional per part, then expanded to the part’s full feature size)
    #     mask = self._generate_mask(ann_list_dict, duration_frames)
        
    #     # Load or create the text embedding tensor
    #     if self.disable:
    #         # When disabled simply return the file path and a duration
    #         dummy = {"x": path, "length": duration_frames}
    #         return dummy, text_dict
        
    #     if path and self.no_model:
    #         try:
    #             if path not in self.texts:
    #                 text_path = os.path.join(self.base_dir, path + ".npz")
    #                 print(f"Loading text embeddings from {text_path}")
    #                 text_file_data = np.load(text_path)
                    
    #                 # Determine the number of frames from one of the arrays in the file
    #                 sample_key = next(iter(text_file_data.files))
    #                 num_frames = text_file_data[sample_key].shape[0]
                    
    #                 arrays = []
    #                 for bp in self.body_part_order:
    #                     if bp in text_file_data.files:
    #                         arrays.append(text_file_data[bp])
    #                     else:
    #                         # If a body part is missing, substitute with zeros and print a warning
    #                         print(f"Warning: '{bp}' not found in {text_path}. Inserting zeros.")
    #                         arrays.append(np.zeros((num_frames, self.features_per_part)))
                    
    #                 # Combine arrays in the order specified by self.body_part_order
    #                 combined_array = np.concatenate(arrays, axis=1)
    #                 self.texts[path] = torch.from_numpy(combined_array).to(torch.float)
                
    #             # Extract the relevant slice of embeddings
    #             begin = int(start * self.fps)
    #             end_frame = int(end * self.fps)
    #             text_data = self.texts[path][begin:end_frame]
    #         except (FileNotFoundError, IOError, KeyError) as e:
    #             print(f"Error loading text embeddings: {e}")
    #             text_data = torch.zeros((duration_frames, self.total_features), dtype=torch.float, device=self.device)

    #     # Sanity check: for each body part, if the mask is 0, the embedding for that part should be unknown.
    #     self._sanity_check(text_data, mask)

    #     # Apply PCA if specified, but before that, enforce that for every frame where mask==0 for a body part
    #     # the corresponding embedding is replaced by a zero vector.
    #     if self.pca > 0 and self.pca_components is not None:
    #         text_data = self._apply_pca(text_data, mask)
        
    #     # Return the embeddings along with their lengths and the mask.
    #     x_dict = {"x": text_data, "length": text_data.shape[0], "mask": mask}
    #     return x_dict, ann_list_dict

    # def _apply_pca(self, text_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply PCA dimension reduction for each body part separately without using a loop.
    #     For masked parts (mask=0), replace with zeros after PCA transformation.
    #     Expects text_data of shape (frames, total_features) and mask of the same shape.
    #     """
    #     frames = text_data.shape[0]
    #     original_features_per_part = self.features_per_part
    #     reduced_dim_per_part = self.pca
    #     reduced_features = reduced_dim_per_part * len(self.body_part_order)
        
    #     # Create the output tensor for reduced data
    #     reduced_data = torch.zeros((frames, reduced_features), dtype=torch.float32, device=self.device)
        
    #     # Convert the entire tensor to numpy for PCA processing
    #     text_data_np = text_data.cpu().numpy()
        
    #     # Process each body part
    #     for i, part in enumerate(self.body_part_order):
    #         # Extract features for current body part
    #         col_start = i * original_features_per_part
    #         col_end = (i + 1) * original_features_per_part
            
    #         # Extract the mask for this part (just need first column since mask is same for all features in a part)
    #         part_mask = mask[:, col_start].cpu().numpy().reshape(-1, 1)  # shape: (frames, 1)
            
    #         # Apply PCA to all frames at once for this body part
    #         part_features = text_data_np[:, col_start:col_end]
    #         transformed = self.pca_components.transform(part_features)
            
    #         # Take only the needed dimensions
    #         transformed = transformed[:, :reduced_dim_per_part]
            
    #         # Apply mask: zero out the rows where mask is 0
    #         # Broadcasting part_mask to match transformed shape
    #         masked_transformed = transformed * part_mask
            
    #         # Store the result in the appropriate slice of reduced_data
    #         reduced_slice = slice(i * reduced_dim_per_part, (i + 1) * reduced_dim_per_part)
    #         reduced_data[:, reduced_slice] = torch.from_numpy(masked_transformed)
        
    #     return reduced_data