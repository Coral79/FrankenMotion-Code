import logging
import torch
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def get_embedding_with_model_fallback(text_encoder, text: str) -> Dict[str, Any]:
    """
    Get embedding for text with fallback to model computation.
    
    Strategy:
    1. Use precomputed embedding if available
    2. Use cached embedding if computed this session
    3. Compute on-the-fly using TextToEmb model
    4. Fall back to 'unknown' if no_model=True
    
    Args:
        text_encoder: TextEmbeddings instance
        text: Text label to get embedding for
        
    Returns:
        Embedding dict with 'x' and 'length' keys
    """
    # Check precomputed embeddings
    if text in text_encoder:
        return text_encoder.get_embedding(text)
    
    # Check session cache
    if text in text_encoder.cache:
        return text_encoder.cache[text]
    
    # Compute on-the-fly if model is available
    if not text_encoder.no_model:
        logger.info(f"Computing embedding for unseen label: '{text}'")
        model = text_encoder.get_model()
        
        # Call model with text - it returns a dict or tensor
        result = model(text)
        
        # Handle different return types
        if isinstance(result, torch.Tensor):
            # Sentence embeddings: shape (1, embedding_dim) or (embedding_dim,)
            # if result.ndim == 2:
            #     x_dict = {"x": result[0], "length": 1}
            # else:
            #     x_dict = {"x": result, "length": 1}
            x_dict = {"x": result, "length": 1}
        elif isinstance(result, dict):
            # Already in correct format
            x_dict = result
        else:
            raise ValueError(f"Unexpected model output type: {type(result)}")
        
        text_encoder.cache[text] = x_dict
        return x_dict
    
    # Fall back to unknown
    logger.warning(f"Label '{text}' not in precomputed embeddings and no_model=True, using 'unknown'")
    return text_encoder.get_embedding("unknown")


def load_from_annotation_with_model(
    text_encoder,
    annotations: List[Dict[str, Any]],
    path: str,
    start: float,
    end: float
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[str, int]]]]:
    """
    Load embeddings from annotations, computing unseen labels on-the-fly.
    
    This is a wrapper around TextEmbeddings that replaces get_embedding calls
    with get_embedding_with_model_fallback to enable on-the-fly computation.
    
    Args:
        text_encoder: TextEmbeddings instance
        annotations: List of annotation dicts
        path: Path identifier
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Tuple of (embedding_dict, text_dict)
    """
    # Get duration in frames
    duration_frames = int(text_encoder.fps * (end - start))
    
    # Format annotations into dict
    ann_list_dict_all = text_encoder._format_annotations(annotations, start, end, text_encoder.fps)
    
    # Apply masking if enabled
    if text_encoder.rand_mask:
        ann_list_dict_part = text_encoder.mask_annotations(ann_list_dict_all)
    else:
        ann_list_dict_part = ann_list_dict_all
    
    # Filter to expected body parts
    ann_list_dict = {
        part: ann_list 
        for part, ann_list in ann_list_dict_part.items() 
        if part in text_encoder.body_part_order
    }
    
    # Generate mask
    mask = text_encoder._generate_mask(ann_list_dict, duration_frames)
    
    # Check if disabled
    if text_encoder.disable:
        dummy = {"x": path, "length": duration_frames}
        return dummy, ann_list_dict
    
    # Create text data tensor
    text_data = torch.zeros(
        (duration_frames, text_encoder.total_features), 
        dtype=torch.float, 
        device=text_encoder.device
    )
    
    # Fill text_data for each body part
    for part_idx, part in enumerate(text_encoder.body_part_order):
        part_start = part_idx * text_encoder.features_per_part
        part_end = (part_idx + 1) * text_encoder.features_per_part
        
        if part in ann_list_dict:
            ann_list = ann_list_dict[part]
            current_frame = 0
            last_embedding_tensor = None
            
            for label, nframes in ann_list:
                # KEY CHANGE: Use fallback function instead of get_embedding
                embedding_dict = get_embedding_with_model_fallback(text_encoder, label)
                
                embedding = embedding_dict["x"]
                if isinstance(embedding, torch.Tensor):
                    embedding_tensor = embedding.float()
                else:
                    import numpy as np
                    embedding_tensor = torch.from_numpy(embedding).float()
                
                last_embedding_tensor = embedding_tensor
                
                end_frame = min(current_frame + nframes, duration_frames)
                frames_to_fill = end_frame - current_frame
                
                text_data[current_frame:end_frame, part_start:part_end] = (
                    embedding_tensor.expand(frames_to_fill, -1)
                )
                
                current_frame = end_frame
                if current_frame >= duration_frames:
                    break
            
            # Pad remaining frames if needed
            if current_frame < duration_frames and last_embedding_tensor is not None:
                remaining_frames = duration_frames - current_frame
                text_data[current_frame:duration_frames, part_start:part_end] = (
                    last_embedding_tensor.unsqueeze(0).expand(remaining_frames, -1)
                )
        else:
            raise ValueError(f"No annotations found for body part '{part}'")
    
    # Convert to device
    text_data = text_data.to(text_encoder.device)
    
    # Apply PCA if specified
    if text_encoder.pca > 0 and text_encoder.pca_components is not None:
        text_data, reduced_mask = text_encoder._apply_pca(text_data, mask)
    else:
        reduced_mask = mask
    
    # Build return dict
    x_dict = {}
    
    # Handle global text if enabled
    if text_encoder.load_global:
        if text_encoder.load_global_type == "action":
            action_list_dict = {
                part: ann_list 
                for part, ann_list in ann_list_dict_all.items() 
                if part == "action"
            }
            if "action" in action_list_dict:
                ann_list = action_list_dict['action']
            else:
                ann_list = []
        else:  # caption
            caption_list_dict = {
                part: ann_list 
                for part, ann_list in ann_list_dict_all.items() 
                if part == 'sequence_caption'
            }
            if 'sequence_caption' in caption_list_dict:
                ann_list = caption_list_dict['sequence_caption']
            else:
                ann_list = []
        
        if ann_list:
            for label, nframes in ann_list:
                global_x_dict = get_embedding_with_model_fallback(text_encoder, label)
                global_x_dict['text'] = label
                x_dict = global_x_dict
                break  # Only use first label
    
    # Add local embeddings
    x_dict['local'] = {
        "x": text_data, 
        "length": text_data.shape[0], 
        "mask": reduced_mask
    }
    
    return x_dict, ann_list_dict