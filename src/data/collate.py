import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate

#the True values in the mask indicate valid positions in the sequence (actual data), 
# while False values indicate padding positions
def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_text_motion_part(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    other_keys = [key for key in keys if key not in ["x", "tx", "text","stats_mask", "motion_dim", "text_dim", "condition"]]
    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    
    # x_motion = x["x"]
    # x_text = x["tx"]["x"]

    x = collate_tensor_with_padding([x["x"] for x in lst_elements])
    if device is not None:
        x = x.to(device)

    batch["x"] = x

    if "length" in batch:
        batch["mask"] = length_to_mask(batch["length"], device=x.device)

    # Handle stats_mask if present
    if "stats_mask" in one_el:
        stats_mask = collate_tensor_with_padding([x["stats_mask"] for x in lst_elements])
        batch["stats_mask"] = stats_mask
    # # text embeddings
    # if "tx" in keys:
    #     assert "x" in one_el["tx"]
    #     assert "length" in one_el["tx"]
    #     tx_x = collate_tensor_with_padding([x["tx"]["x"] for x in lst_elements])
    #     tx_length = default_collate([x["tx"]["length"] for x in lst_elements])
    #     tx_mask = length_to_mask(tx_length, device=tx_x.device)
    #     batch["tx"] = {"x": tx_x, "length": tx_length, "mask": tx_mask}

    if "tx" in keys:
        if "local" in one_el["tx"].keys():
            # Validate required tx fields
            assert "x" in one_el["tx"]["local"], "Missing 'x' field in tx dictionary"
            assert "mask" in one_el["tx"]["local"], "Missing 'mask' field in tx dictionary"
            
            # Collate tx tensors with padding
            tx_mask = collate_tensor_with_padding([x["tx"]["local"]["mask"] for x in lst_elements])
            tx_x = collate_tensor_with_padding([x["tx"]["local"]["x"] for x in lst_elements])
            
            # Initialize the batch tx dictionary with required fields
            batch_txf = {"x": tx_x, "mask": tx_mask}
            
            # Handle inpainting mask conditionally
            if "condition" in keys:
                if one_el.get("condition") == 't2m':
                    motion_mask_width = one_el["x"].shape[1] - one_el["tx"]["local"]["x"].shape[1]
                    inpainting_motion_mask = torch.ones((tx_x.shape[1], motion_mask_width))
                    inpainting_text_mask = torch.zeros((tx_x.shape[1], one_el["tx"]["local"]["x"].shape[1]))
                    inpainting_mask = torch.cat([inpainting_motion_mask, inpainting_text_mask], dim=1)
                    batch["inpainting_mask"] = inpainting_mask
                elif one_el.get("condition") == 'm2t':
                    motion_mask_width = one_el["x"].shape[1] - one_el["tx"]["local"]["x"].shape[1]
                    inpainting_motion_mask = torch.zeros((tx_x.shape[1], motion_mask_width))
                    inpainting_text_mask = torch.ones((tx_x.shape[1], one_el["tx"]["local"]["x"].shape[1]))
                    inpainting_mask = torch.cat([inpainting_motion_mask, inpainting_text_mask], dim=1)
                    batch["inpainting_mask"] = inpainting_mask
            
            # Add tx dictionary to batch
            batch["txf"] = batch_txf

        if "x" in one_el["tx"].keys():
            # Validate required tx fields
            assert "x" in one_el["tx"], "Missing 'x' field in tx dictionary"
            assert "length" in one_el["tx"], "Missing 'mask' field in tx dictionary"     
            # Collate tx tensors with padding
            tx_length_g = default_collate([x["tx"]["length"] for x in lst_elements])
            tx_x_g = collate_tensor_with_padding([x["tx"]["x"] for x in lst_elements])
            tx_mask_g = length_to_mask(tx_length_g, device=tx_x.device)
            
            # Initialize the batch tx dictionary with required fields
            batch_tx = {"x": tx_x_g, "length": tx_length_g, "mask": tx_mask_g}
            batch["tx"] = batch_tx
            batch['global_text'] = default_collate([x["tx"]["text"] for x in lst_elements])
            
        else:
            batch["tx"] = batch_txf



    batch["text"] = [x["text"] for x in lst_elements]


    if "tx_uncond" in keys:
        # only one is enough
        batch["tx_uncond"] = one_el["tx_uncond"]
        tx_uncond_x = default_collate([x["tx_uncond"]["x"] for x in lst_elements])
        tx_uncond_length = default_collate([x["tx_uncond"]["length"] for x in lst_elements])
        batch["tx_uncond_batch"] = {"x": tx_uncond_x, "length": tx_uncond_length}

    if "motion_dim" in keys:
        batch["motion_dim"] = one_el["motion_dim"]

    if "text_dim" in keys:
        batch["text_dim"] = one_el["text_dim"]

    return batch
