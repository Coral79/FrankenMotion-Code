import os
import torch.nn as nn
import torch
from torch import Tensor
from typing import Dict, List
import torch.nn.functional as F
# [新增引用] 请确保这些在 import 列表里
from typing import Optional
try:
    from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
except ImportError:
    print("Warning: transformers not installed.")
    

class CLIP_wrapper(nn.Module):
    def __init__(self, modelname: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device

        import clip

        model, preprocess = clip.load(modelname, device)
        self.tokenizer = clip.tokenize
        self.clip_model = model.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device
        tokens = self.tokenizer(texts, truncate=True).to(device)
        return self.clip_model.encode_text(tokens).float()

# [新增类] Qwen Wrapper (默认截断为 512 维)
class Qwen_wrapper(nn.Module):
    def __init__(self, modelname: str, qwen_dim: int = 512, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.qwen_dim = qwen_dim
        
        print(f"Loading Qwen model: {modelname} on {device}...")
        print(f"👉 Output dimension forced to: {qwen_dim}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(modelname, trust_remote_code=True).to(device)
        self.model.eval()
        
        for p in self.model.parameters():
            p.requires_grad = False
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, mode: bool = True) -> nn.Module:
        self.training = False
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Tensor:
        device = device if device is not None else self.device
        
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=77
        ).to(device)
        
        outputs = self.model(**inputs)
        
        # Mean Pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Slicing (Dimension truncation to 512)
        if self.qwen_dim is not None:
            embeddings = embeddings[:, :self.qwen_dim]
            
        return F.normalize(embeddings, p=2, dim=1)

# [新增类] T5 Wrapper
class T5_wrapper(nn.Module):
    def __init__(self, modelname: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"Loading T5 model: {modelname} on {device}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(modelname)
        self.model = T5EncoderModel.from_pretrained(modelname).to(device)
        self.model.eval()
        
        for p in self.model.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> nn.Module:
        self.training = False
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Tensor:
        device = device if device is not None else self.device
        
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=77
        ).to(device)
        
        outputs = self.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        
        # Weighted Mean Pooling
        token_embeddings = outputs.last_hidden_state
        attn_mask = inputs["attention_mask"]
        summed = torch.sum(token_embeddings * attn_mask.unsqueeze(-1), dim=1)
        counts = attn_mask.sum(dim=1, keepdim=True)
        pooled_sentence = summed / counts
        
        return F.normalize(pooled_sentence, p=2, dim=1)
    
    
class HF_wrapper(nn.Module):
    def __init__(
        self, modelpath: str, mean_pooling: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = device

        from transformers import AutoTokenizer, AutoModel, T5EncoderModel
        from transformers import logging

        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        if modelpath == "google/flan-t5-xl":
            # only load the encoder not the decoder as well
            self.text_model = T5EncoderModel.from_pretrained(modelpath)
        else:
            # Text model
            self.text_model = AutoModel.from_pretrained(modelpath)
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

        if mean_pooling:
            self.forward = self.forward_pooling

        # put it in eval mode by default
        self.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model.encoder(**encoded_inputs.to(device))
        length = encoded_inputs.attention_mask.to(dtype=bool).sum(1)

        if squeeze:
            x_dict = {"x": output.last_hidden_state.detach()[0], "length": length[0]}
        else:
            x_dict = {"x": output.last_hidden_state.detach(), "length": length}
        return x_dict

    @torch.no_grad()
    def forward_pooling(self, texts: List[str], device=None) -> Tensor:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        # From: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(device))
        attention_mask = encoded_inputs["attention_mask"]

        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = output["last_hidden_state"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        if squeeze:
            sentence_embeddings = sentence_embeddings[0]
        return sentence_embeddings


def TextToEmb(modelpath: str, mean_pooling: bool = False, device: str = "cpu"):
    # if modelpath == "clip":
    if "clip" in modelpath:
        modelpath = "ViT-B/32"

    # clip models
    if modelpath in [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ]:
        return CLIP_wrapper(modelpath, device=device)
    # hugging face
    # 2. [新增] Qwen Models
    elif "qwen" in modelpath.lower():
        # 默认使用 512 维，与 CLIP 对齐
        return Qwen_wrapper(modelpath, qwen_dim=512, device=device)
    
    # 3. [新增] T5 Models
    elif "t5" in modelpath.lower():
        return T5_wrapper(modelpath, device=device)
    
    else:
        return HF_wrapper(modelpath, mean_pooling, device)
