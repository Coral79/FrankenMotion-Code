import os
import sys
import numpy as np
import json
import torch
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pathlib import Path
import orjson
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

def load_json(json_path):
    """Load JSON data from file using orjson for performance."""
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())

def predict_frame_txt(local_embed, clip_dir, pca_dim=None, method='KNN'):
    """
    Predict the text descriptions for generated motion sequences using K-Nearest Neighbors.
    
    Args:
        local_embed: The embedded text features from the model's output
        clip_dir: Directory containing clip embeddings, index, and slice files
        pca_dim: Dimension per body part after PCA reduction (if None, assumes no PCA)
        method: The method to use for prediction (default: 'KNN')
        
    Returns:
        idx_all: List of indices for predicted texts
        text_all: List of predicted text descriptions
    """
    # Path to required files
    clip_path = os.path.join(clip_dir, "clip.npy")
    index_path = os.path.join(clip_dir, "clip_index.json")
    
    # Load embeddings and index
    try:
        embeddings_np = np.load(clip_path)
        texts_dict = load_json(index_path)
        inverse_texts_dict = {v: k for k, v in texts_dict.items()}
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise ValueError(f"Failed to load embeddings from {clip_dir}: {e}")
    
    logger.info(f"Calculating Nearest Neighbour in {'PCA' if pca_dim else 'original'} space")
    
    # Parse shape of input embedding
    num_seq, num_frame, num_feats = np.shape(local_embed)
    pred_embed = np.transpose(local_embed, (0, 2, 1))  # Shape: (num_seq, num_feats, num_frame)
    
    # If we have PCA-reduced data, fit PCA on reference embeddings
    if pca_dim is not None and pca_dim > 0:
        # Create PCA model and fit to reference embeddings
        pca = PCA(n_components=pca_dim)
        pca.fit(embeddings_np)
        
        # Apply PCA to reference embeddings
        embeddings_np = pca.transform(embeddings_np)
        logger.info(f"Applied PCA to reference embeddings: {embeddings_np.shape}")
    
    # Initialize and train KNN
    if method == "KNN":
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(embeddings_np)
    else:
        logger.error("We haven't implemented other ways for prediction")
        raise ValueError("Unsupported prediction method")
    
    text_all = []
    idx_all = []
    
    for i in range(num_seq):
        # KNN Search
        distances, indices = knn.kneighbors(pred_embed[i])  # Shape: (num_frame, 1)
        
        text_seq = []
        idx_seq = []
        for clip_idx, neighbors in enumerate(indices):
            for neighbor_idx in neighbors:
                text_seq.append(inverse_texts_dict[neighbor_idx])
                idx_seq.append(neighbor_idx)
        
        text_all.append(text_seq)
        idx_all.append(idx_seq)
    
    return idx_all, text_all

def compress_text_predictions(text_list):
    """
    Compress a list of text predictions into (label, count) tuples.
    
    Args:
        text_list: List of predicted texts
        
    Returns:
        List of (label, count) tuples
    """
    if not text_list:
        return []
    
    compressed_list = []
    current_label = text_list[0]
    current_count = 1
    
    # Process from the second frame onwards
    for i in range(1, len(text_list)):
        if text_list[i] == current_label:
            # Same label, increment count
            current_count += 1
        else:
            # Different label, add the current run and start a new one
            compressed_list.append((current_label, current_count))
            current_label = text_list[i]
            current_count = 1
    
    # Add the last run
    compressed_list.append((current_label, current_count))
    
    return compressed_list

def extract_texts(xstart, length, motion_dim, text_dim, clip_dir, 
                  body_parts=None, output_path=None, method='KNN', pca_dim=None):
    if body_parts is None:
        body_parts = ["trajectory", "spine", "left_arm", "right_arm", "left_leg", "right_leg"]
    
    num_parts = len(body_parts)
    features_per_part = text_dim // num_parts

    # Sanity check on dimension
    if xstart.shape[1] < motion_dim + text_dim:
        logger.warning(f"Unexpected shape: {xstart.shape}. Expecting at least {motion_dim + text_dim} features.")
        return {}

    # Extract text embeddings: (length, text_dim)
    text_embeddings_full = xstart[:length, motion_dim:motion_dim + text_dim].cpu().numpy()

    ann_list_dict = {}

    for i, part in enumerate(body_parts):
        start_idx = i * features_per_part
        end_idx = (i + 1) * features_per_part

        part_embed = text_embeddings_full[:, start_idx:end_idx]  # (frames, features_per_part)
        part_embed = np.expand_dims(part_embed, axis=0)  # (1, frames, features_per_part)

        # Transpose to (1, features, frames) to match expected input
        part_embed = np.transpose(part_embed, (0, 2, 1))

        try:
            pred_indices, pred_texts = predict_frame_txt(
                part_embed,
                clip_dir,
                pca_dim=pca_dim,
                method=method
            )
        except Exception as e:
            logger.error(f"Prediction failed for body part '{part}': {e}")
            ann_list_dict[part] = [("unknown", length)]
            continue

        part_texts = pred_texts[0] if pred_texts else ["unknown"] * length

        if len(part_texts) < length:
            part_texts += ["unknown"] * (length - len(part_texts))

        compressed = compress_text_predictions(part_texts)
        ann_list_dict[part] = compressed

    # Save output if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write("Predicted text annotations:\n\n")
            for part, compressed_list in ann_list_dict.items():
                f.write(f"{part}:\n")
                for label, count in compressed_list:
                    f.write(f"  {label} ({count} frames)\n")
                f.write("\n")
        logger.info(f"Saved predicted text to {output_path}")

    return ann_list_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from generated sequences")
    parser.add_argument("--npy_file", type=str, help="Path to a generated npy file")
    parser.add_argument("--clip_dir", type=str, required=True, help="Directory containing clip.npy, clip_index.json, and clip_slice.npy")
    parser.add_argument("--motion_dim", type=int, required=True, help="Dimension of motion features")
    parser.add_argument("--text_dim", type=int, required=True, help="Dimension of text features")
    parser.add_argument("--output", type=str, help="Output path for predicted text")
    parser.add_argument("--method", type=str, default="KNN", help="Method for prediction (default: KNN)")
    parser.add_argument("--body_parts", nargs='+', default=["trajectory", "spine", "left_arm", "right_arm", "left_leg", "right_leg"], 
                        help="List of body part names")
    parser.add_argument("--pca_dim", type=int, help="Dimension per body part after PCA reduction (if not specified, tries to infer)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.npy_file:
        # Process a single file
        try:
            # Load the npy file
            data = np.load(args.npy_file)
            
            # Get length
            length = data.shape[0]
            
            # Extract text
            ann_list_dict = extract_texts(
                xstart=data,
                length=length,
                motion_dim=args.motion_dim,
                text_dim=args.text_dim,
                clip_dir=args.clip_dir,
                body_parts=args.body_parts,
                output_path=args.output or args.npy_file.replace('.npy', '_predicted_text.txt'),
                method=args.method,
                pca_dim=args.pca_dim
            )
            
            if ann_list_dict:
                logger.info(f"Successfully extracted text predictions for {args.npy_file}")
                
                # Print a summary
                for part, compressed_list in ann_list_dict.items():
                    logger.info(f"{part}: {len(compressed_list)} segments")
            else:
                logger.warning(f"No text predictions generated for {args.npy_file}")
        except Exception as e:
            logger.error(f"Error processing {args.npy_file}: {e}")
            sys.exit(1)
    else:
        logger.error("--npy_file must be provided")
        sys.exit(1)