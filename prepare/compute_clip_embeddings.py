"""
Compute CLIP ViT-B/32 text embeddings for the Frankenstein Dataset.

Pre-computed embeddings are already provided in the dataset download.
Use this script only if you want to recompute them yourself.

Usage:
    pip install clip
    python prepare/compute_clip_embeddings.py --dataset_dir datasets/annotations/frankenstein-dataset
"""

import argparse
import json
import os

import clip
import numpy as np
import torch
from tqdm import tqdm


def compute_clip_embeddings(dataset_dir: str, device: str = "cuda"):
    annotations_path = os.path.join(dataset_dir, "annotations", "annotations.json")
    output_dir = os.path.join(dataset_dir, "text_embeddings", "clip")
    os.makedirs(output_dir, exist_ok=True)

    with open(annotations_path) as f:
        annotations = json.load(f)

    # Collect all unique texts
    all_texts = [""]
    for entry in annotations.values():
        for ann in entry["annotations"]:
            all_texts.append(ann["text"])
    all_texts = list(set(all_texts))
    print(f"Computing embeddings for {len(all_texts)} unique texts...")

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    batch_size = 256
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), batch_size)):
            batch = all_texts[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            embeddings = model.encode_text(tokens).float().cpu()
            all_embeddings.append(embeddings)

    embeddings_matrix = torch.cat(all_embeddings).numpy()  # [N, 512]

    # clip_slice.npy: [[begin, end], ...] — each text maps to one embedding row
    slices = np.array([[i, i + 1] for i in range(len(all_texts))], dtype=np.int64)

    # clip_index.json: {text: index}
    index = {text: i for i, text in enumerate(all_texts)}

    np.save(os.path.join(output_dir, "clip.npy"), embeddings_matrix)
    np.save(os.path.join(output_dir, "clip_slice.npy"), slices)
    with open(os.path.join(output_dir, "clip_index.json"), "w") as f:
        json.dump(index, f)

    print(f"Saved embeddings to {output_dir}")
    print(f"  clip.npy:        {embeddings_matrix.shape}")
    print(f"  clip_slice.npy:  {slices.shape}")
    print(f"  clip_index.json: {len(index)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the downloaded frankenstein-dataset directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    compute_clip_embeddings(args.dataset_dir, args.device)
