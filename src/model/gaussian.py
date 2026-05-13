import logging
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion.diffusion_base import DiffuserBase
from .diffusion.discrete_diffusion import DiscreteDiffusion
from ..data.collate import length_to_mask, collate_tensor_with_padding
from src.stmc import combine_features_intervals, interpolate_intervals
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from pytorch_lightning import LightningModule
from einops import rearrange, repeat, reduce
from omegaconf import DictConfig
from typing import Optional, Dict, Any, Tuple
import numpy as np
# from ..model.time_sampler import get_time_sampler, TimeSamplerCfg
# from ..model.time_sampler.independent import IndependentCfg  
# from ..model.time_sampler.shared import SharedCfg
# from ..model.time_sampler.mean_beta import MeanBetaCfg


class LearnableMSELoss(nn.Module):
    """
    A loss function that combines two MSE losses with a learnable weight parameter.
    
    This loss function computes loss = MSE(m, m_pred) + lambda * MSE(f, f_pred),
    where lambda is either a fixed or learnable parameter.
    
    Args:
        initial_lambda (float): Initial value for the lambda weight (default: 1.0)
        learn_lambda (bool): Whether lambda should be learnable (default: True)
    """
    def __init__(self, initial_lambda=1.0, learn_lambda=True):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        if learn_lambda:
            self.log_lambda = nn.Parameter(torch.tensor(float(initial_lambda)).log())
        else:
            self.register_buffer("log_lambda", torch.tensor(float(initial_lambda)).log())
    
    def forward(self, m, m_pred, f, f_pred):
        lambda_val = torch.exp(self.log_lambda)
        loss_m = self.mse(m, m_pred)
        loss_f = self.mse(f, f_pred)
        total_loss = loss_m + lambda_val * loss_f
        
        return total_loss, loss_m, loss_f, lambda_val

class WeightedMSELoss(nn.Module):
    """
    A weighted MSE loss function with emphasis on root vector (first 4 dimensions).
    
    Args:
        root_weight (float): Weight multiplier for root vector dimensions (default: 2.0)
        other_weight (float): Weight multiplier for other dimensions (default: 1.0)
        reduction (str): Reduction method ('mean', 'sum', 'none', default: 'mean')
    """
    def __init__(self, root_weight=3.0, other_weight=1.0, reduction='mean'):
        super().__init__()
        self.root_weight = root_weight
        self.other_weight = other_weight
        self.reduction = reduction
    
    def forward(self, pred, target, weights=None):
        # Convert inputs to float
        pred = pred.float()
        target = target.float()
        
        # Create automatic weights for root vector emphasis if not provided
        if weights is None:
            weights = torch.ones_like(pred)
            # Emphasize first 4 dimensions (root vector)
            weights[..., :4] = self.root_weight
            weights[..., 4:] = self.other_weight
        else:
            weights = weights.float()
        
        # Compute weighted squared differences
        squared_diff = (pred - target) ** 2
        weighted_squared_diff = weights * squared_diff
        
        # Apply reduction
        if self.reduction == 'none':
            return weighted_squared_diff
        elif self.reduction == 'sum':
            return weighted_squared_diff.sum()
        elif self.reduction == 'mean':
            return torch.mean(
                weighted_squared_diff.reshape(target.shape[0], -1),
                dim=1
            ).mean()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class LearnableWeightedMSELoss(nn.Module):
    """
    A loss function that combines weighted MSE losses for different components with a learnable weight.
    
    Args:
        initial_lambda (float): Initial value for the lambda weight (default: 1.0)
        learn_lambda (bool): Whether lambda should be learnable (default: True)
        reduction (str): Reduction method ('mean', 'sum', 'none', default: 'mean')
    """
    def __init__(self, initial_lambda=1.0, learn_lambda=True, reduction='mean'):
        super().__init__()
        self.weighted_mse = WeightedMSELoss(reduction=reduction)
        
        if learn_lambda:
            self.log_lambda = nn.Parameter(torch.tensor(float(initial_lambda)).log())
        else:
            self.register_buffer("log_lambda", torch.tensor(float(initial_lambda)).log())
    
    def forward(self, motion_pred, motion_target, text_pred, text_target, weights=None):
        lambda_val = torch.exp(self.log_lambda)
        
        # Calculate weighted MSE for motion and text components
        loss_m = self.weighted_mse(motion_pred, motion_target, weights)
        loss_t = self.weighted_mse(text_pred, text_target, weights)
        
        # Combine losses with learnable lambda
        total_loss = loss_m + lambda_val * loss_t
        
        return total_loss, loss_m, loss_t, lambda_val

# Inplace operator: return the original tensor
# work with a list of tensor as well
def masked(tensor, mask): #set False to 0
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = 0.0
    return tensor

def extract(a, t, x_shape):
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


logger = logging.getLogger(__name__)


def compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=1, logit_mean=0.0, logit_std=1.0, mode_scale=1.29):
    """
    Sample timesteps non-uniformly based on the specified weighting scheme.
    Ported from SD3 DreamBooth training code.
    """
    if weighting_scheme == "logit_normal":
        # Sample from logit-normal distribution
        u = torch.randn(batch_size).mul_(logit_std).add_(logit_mean)
        u = torch.sigmoid(u)
    elif weighting_scheme == "mode":
        # Sample more densely near a specific timestep range (mode)
        u = torch.rand(batch_size) * mode_scale
        u = u - (mode_scale - 1) / 2
        u = u.clamp(0, 1)
    elif weighting_scheme == "cosmap":
        # Cosine mapping
        u = torch.rand(batch_size)
        u = torch.cos(u * torch.pi / 2)
    else:  # "sigma_sqrt" or default
        # Square root sampling favors lower noise levels
        u = torch.sqrt(torch.rand(batch_size))
    
    return u

def compute_loss_weighting_for_flow_matching(weighting_scheme="logit_normal", sigmas=None):
    """
    Compute loss weighting for different timesteps in flow matching.
    Adapted from SD3 DreamBooth training code.
    """
    if weighting_scheme == "logit_normal":
        # Loss weighting proportional to sigma
        return sigmas
    elif weighting_scheme == "mode":
        # Increased weighting near a specific timestep range
        return 1.0 / (1.0 + sigmas)
    elif weighting_scheme == "cosmap":
        # Weighting based on cosine mapping
        return torch.cos(sigmas * torch.pi / 2)
    else:  # "sigma_sqrt" or default
        # Square root weighting
        return torch.sqrt(sigmas)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



class GaussianDiffusion(DiffuserBase):
    name = "gaussian"

    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
        lr: float = 2e-4,
        g_text: bool = True,
        uni: bool = False,
        loss_type: str = 'MSE',
        root_weight: float = 5.0,
        cfg_strategy: str = 'simple',  
    ):
        super().__init__(schedule, timesteps)

        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.lr = lr
        self.prediction = prediction
        self.uni = uni
        self.loss_type = loss_type
        self.cfg_strategy = cfg_strategy

        if self.loss_type == 'MSE':
            self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")
        elif self.loss_type == 'WeightedMSE':
            self.reconstruction_loss = WeightedMSELoss(reduction="mean", root_weight=root_weight)
        elif self.loss_type == 'LearnableMSE':
            self.reconstruction_loss = LearnableMSELoss(initial_lambda=1.0, learn_lambda=True)
        elif self.loss_type == 'LearnableWeightedMSE':
            self.reconstruction_loss = LearnableWeightedMSELoss(initial_lambda=1.0, learn_lambda=True)
        elif self.loss_type == 'SNR_MSE':
            print("special loss for diffusion forcing")
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented")
            

        # if self.uni:
        #     self.reconstruction_loss = LearnableMSELoss(initial_lambda=1.0, learn_lambda=True)
        # else:
        #     self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")
        # self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")

        # normalization
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer
        self.g_text = g_text

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def prepare_tx_emb(self, tx_emb):
        # Text embedding normalization
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        tx = {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"].bool()),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }
        return tx

    def diffusion_step(self, batch, batch_idx, training=False):
        mask = batch["mask"]
        # motion_dim = batch["motion_dim"]
        # text_dim = batch["text_dim"]  
        # Check if we have mixed motion+text data
        has_text = "motion_dim" in batch and "text_dim" in batch
        if has_text:
            motion_dim = batch["motion_dim"]
            text_dim = batch["text_dim"]      

        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask) 
        if "stats_mask" in batch.keys():
            x = x*batch["stats_mask"]

        if self.g_text:
            y = {
                "length": batch["length"],
                "mask": mask,
                "tx": self.prepare_tx_emb(batch["tx"]),
                # the condition is already dropped sometimes in the dataloader
            }
        else:
            y = {
                "length": batch["length"],
                "mask": mask,
                # the condition is already dropped sometimes in the dataloader
            }

        bs = len(x)
        # Sample a diffusion step between 0 and T-1
        # 0 corresponds to noising from x0 to x1
        # T-1 corresponds to noising from xT-1 to xT
        t = torch.randint(0, self.timesteps, (bs,), device=x.device)

        # Create a noisy version of x
        # no noise for padded region
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample(xstart=x, t=t, noise=noise) #masked noise
        xt = masked(xt, mask)
        if "inpainting_mask" in batch.keys():
            self.inpainting_mask = batch["inpainting_mask"].unsqueeze(0)
            xt = self.inpainting_mask*xt + (1-self.inpainting_mask)*x


        # denoise it
        # no drop cond -> this is done in the training dataloader already
        # give "" instead of the text
        # denoise it
        output = masked(self.denoiser(xt, y, t), mask)

        # Predictions
        xstart = masked(self.output_to("x", output, xt, t), mask)
        
        if "inpainting_mask" in batch.keys():
            target = x*self.inpainting_mask
            model_pred = xstart*self.inpainting_mask
        else:
            target = x
            model_pred = xstart

        if self.loss_type == "LearnableMSE" or self.loss_type == "LearnableWeightedMSE":
            # Set up common parameters
            args = [
                target[:,:,:motion_dim], 
                model_pred[:,:,:motion_dim], 
                target[:,:,-text_dim:], 
                model_pred[:,:,-text_dim:]
            ]
            
            xloss, loss_m, loss_f, lambda_val = self.reconstruction_loss(*args)
            
            loss = {
                "loss": xloss,
                "motion loss": loss_m,
                "text loss": loss_f,
                "lambda": lambda_val,
            }
        else:
            xloss = self.reconstruction_loss(target, model_pred)
            loss = {"loss": xloss}
        return loss

    def diffusion_step_uni(self, batch, batch_idx, training=False):
        mask = batch["mask"]
        motion_dim = batch["motion_dim"]
        text_dim = batch["text_dim"]

        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask) 
        if "stats_mask" in batch.keys():
            x = x*batch["stats_mask"]

        if self.g_text:
            y = {
                "length": batch["length"],
                "mask": mask,
                "tx": self.prepare_tx_emb(batch["tx"]),
                # the condition is already dropped sometimes in the dataloader
            }
        else:
            y = {
                "length": batch["length"],
                "mask": mask,
                # the condition is already dropped sometimes in the dataloader
            }

        bs = len(x)
        # Sample a diffusion step between 0 and T-1
        # 0 corresponds to noising from x0 to x1
        # T-1 corresponds to noising from xT-1 to xT
        tm = torch.randint(0, self.timesteps, (bs,), device=x.device)
        tf = torch.randint(0, self.timesteps, (bs,), device=x.device)

        # Create a noisy version of x
        # no noise for padded region
        xm = x[:,:,:motion_dim]
        xf = x[:,:,-text_dim:]
        noisem = masked(torch.randn_like(xm), mask)
        noisef = masked(torch.randn_like(xf), mask)
        xtm = self.q_sample(xstart=xm, t=tm, noise=noisem) #masked noise
        xtm = masked(xtm, mask)
        xtf = self.q_sample(xstart=xf, t=tf, noise=noisef) #masked noise
        xtf = masked(xtf, mask)
        xt = torch.cat([xtm, xtf], dim=2)
        # if "inpainting_mask" in batch["tx"].keys():
        #     self.inpainting_mask = batch["tx"]["inpainting_mask"].unsqueeze(0)
        #     xt = self.inpainting_mask*xtm + (1-self.inpainting_mask)*xtf

        # denoise it
        # no drop cond -> this is done in the training dataloader already
        # give "" instead of the text
        # denoise it
        output = masked(self.denoiser(xt, y, tm, tf), mask)

        outputm = output[:,:,:motion_dim]
        outputf = output[:,:,-text_dim:]

        # Predictions
        xstartm = masked(self.output_to("x", outputm, xtm, tm), mask)
        xstartf = masked(self.output_to("x", outputf, xtf, tf), mask)
        
        # if "inpainting_mask" in batch["tx"].keys():
        #     xstart = self.inpainting_mask*xstartm + (1-self.inpainting_mask)*xstartf
        # else:
        #     xstart = xstartm
        if "stats_mask" in batch.keys():
            xstart = torch.cat([xstartm, xstartf], dim=2)*batch["stats_mask"]
        else:
            xstart = torch.cat([xstartm, xstartf], dim=2)

        xstartm = xstart[:,:,:motion_dim]
        xstartf = xstart[:,:,-text_dim:]

        xloss, loss_m, loss_f, lambda_val = self.reconstruction_loss(xstartm, xm, xstartf, xf)
        loss = {"loss": xloss,
                "motion loss":loss_m,
                "text loss":loss_f,
                "lambda":lambda_val,
        }
        return loss

    def training_step(self, batch, batch_idx):
        import time
        if not hasattr(self, "_prev_end"):  # first batch
            self._prev_end = time.time()
            self._waits, self._steps = [], []
        start = time.time()
        data_wait = start - self._prev_end    
            
            
        bs = len(batch["x"])
        if self.uni:
            loss = self.diffusion_step_uni(batch, batch_idx, training=True)
        else:
            loss = self.diffusion_step(batch, batch_idx, training=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - start
        self._prev_end = time.time()

        # record times
        self._waits.append(data_wait)
        self._steps.append(step_time)

        # print occasionally
        if batch_idx % 50 == 0 or batch_idx == 0:
            avg_wait = sum(self._waits) / len(self._waits)
            avg_step = sum(self._steps) / len(self._steps)
            # print(f"[{batch_idx}] wait={data_wait:.3f}s, step={step_time:.3f}s | avg_wait={avg_wait:.3f}s, avg_step={avg_step:.3f}s")

        
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
            
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        bs = len(batch["x"])
        if self.uni:
            loss = self.diffusion_step_uni(batch, batch_idx, training=True)
        else:
            loss = self.diffusion_step(batch, batch_idx, training=True)
        # loss = self.diffusion_step(batch, batch_idx)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )

        return loss["loss"]

    def on_train_epoch_end(self):
        dico = {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.global_step),
        }
        # reset losses
        self._saved_losses = defaultdict(list)
        self.losses = []
        self.log_dict(dico)

    # dispatch
    def forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        if "timeline" in infos:
            ff = self.stmc_forward
            if "baseline" in infos:
                if "sinc" in infos["baseline"]:
                    ff = self.sinc_baseline
            # for the other baselines stmc handle it
            # STMC generalize one text forward and DiffCollage
        else:
            ff = self.text_forward
        return ff(tx_emb, tx_emb_uncond, infos, progress_bar=progress_bar)

    def text_forward(
        self,
        tx_emb,
        tx_emb_uncond,
        infos,
        progress_bar=tqdm,
    ):
        # normalize text embeddings first
        device = self.device

        lengths = infos["all_lengths"]
        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        duration = max(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, duration, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def batch_forward(
        self,
        batch,
        infos=None,
        progress_bar=tqdm,
    ):
        """
        Performs forward diffusion using data directly from a dataloader batch.
        Uses the same batch format as training_step and diffusion_step.
        
        Args:
            batch: A batch from the dataloader containing motion and text data
            infos: Additional information dictionary (optional)
            progress_bar: Progress bar function (default: tqdm)
            
        Returns:
            Generated motion sequence
        """
        device = self.device
        
        mask = batch["mask"]
        self.motion_dim = batch["motion_dim"] if "motion_dim" in batch else batch["x"].size(2)
        self.text_dim = batch["text_dim"] if "text_dim" in batch else 0


        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask) 
        if "stats_mask" in batch.keys():
            x = x*batch["stats_mask"]

        if self.g_text:
            y = {
                "length": batch["length"],
                "mask": mask,
                "tx": self.prepare_tx_emb(batch["tx"]),
                "infos":infos,
                "tx_uncond": self.prepare_tx_emb(batch["tx_uncond_batch"]), ##start from here to debug##
                # the condition is already dropped sometimes in the dataloader
            }
        else:
            y = {
                "length": batch["length"],
                "mask": mask,
                "infos":infos,
                # the condition is already dropped sometimes in the dataloader
            }

        bs = len(x)
        # debug=True

        # Create a noisy version of x
        # no noise for padded region
        xt = masked(torch.randn_like(x), mask)

        # Diffusion sampling loop
        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            if "inpainting_mask" in batch.keys():
                self.inpainting_mask = batch["inpainting_mask"].unsqueeze(0)
                xt = self.inpainting_mask*xt + (1-self.inpainting_mask)*x
            if self.uni:
                tm = torch.full((bs,), diffusion_step, device=device)
                tf = torch.full((bs,), diffusion_step, device=device)
                if self.inpainting_mask[0,0,:self.motion_dim][0] == 0:
                    tm = torch.full((bs,), 0, device=device)
                if self.inpainting_mask[0,0,-self.text_dim:][0] == 0:
                    tf = torch.full((bs,), 0, device=device)
                xt, xstart = self.p_sample_uni(xt, y, tm, tf)
            else:
                t = torch.full((bs,), diffusion_step, device=device)
                xt, xstart = self.p_sample(xt, y, t)

            # Ensure we respect the mask at each step
            xt = masked(xt, mask)
            # if debug and diffusion_step == 70:
            #     xstart = self.motion_normalizer.inverse(xstart)
            #     return xstart

        # Denormalize the output
        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def _prepare_xt_for_cfg(self, xt):
        """
        Prepare the input xt for the unconditional branch based on CFG strategy.
        
        Args:
            xt: noisy input tensor
            cfg_strategy: strategy for preparing unconditional input
                - 'simple': mask out conditioning part with zeros (if inpainting_mask exists)
                - 'full_noise': replace conditioning part with fresh noise
                - 'standard': use original xt without modification
                - 'partial': partially mask conditioning (weighted combination)
        
        Returns:
            xt_uncond: modified input for unconditional denoiser
        """
        if self.cfg_strategy == 'standard':
            # Standard CFG: use same xt for both branches
            return xt
        
        elif self.cfg_strategy == 'simple':
            # Simple CFG: mask out conditioning part with zeros
            if hasattr(self, 'inpainting_mask') and self.inpainting_mask is not None:
                return self.inpainting_mask * xt + (1 - self.inpainting_mask) * 0
            return xt
        
        elif self.cfg_strategy == 'full_noise':
            # Full noise CFG: replace conditioning part with fresh noise
            if hasattr(self, 'inpainting_mask') and self.inpainting_mask is not None:
                noise = torch.randn_like(xt)
                return self.inpainting_mask * xt + (1 - self.inpainting_mask) * noise
            return xt
        
        elif self.cfg_strategy == 'partial':
            # Partial masking: weighted combination (e.g., 50% original, 50% zero)
            if hasattr(self, 'inpainting_mask') and self.inpainting_mask is not None:
                alpha = 0.5  # can be made configurable
                masked_part = (1 - self.inpainting_mask) * xt * alpha
                return self.inpainting_mask * xt + masked_part
            return xt
        
        else:
            raise ValueError(f"Unknown CFG strategy: {self.cfg_strategy}")
    
    def p_sample(self, xt, y, t):
        # guided forward
        output_cond = self.denoiser(xt, y, t)

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()  # not a deep copy
            # Only set tx_uncond if it exists
            if "tx_uncond" in y_uncond:
                y_uncond["tx"] = y_uncond["tx_uncond"]

            xt_uncond = self._prepare_xt_for_cfg(xt)
            output_uncond = self.denoiser(xt_uncond, y_uncond, t)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart

    def p_sample_uni(self, xt, y, tm, tf):
        # guided forward
        output_cond = self.denoiser(xt, y, tm, tf)

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xt, y_uncond, t)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        xtm = xt[:,:,:self.motion_dim]
        xtf = xt[:,:,-self.text_dim:]
        outputm = output[:,:,:self.motion_dim]
        outputf = output[:,:,-self.text_dim:]
        meanm, sigmam = self.q_posterior_distribution_from_output_and_xt(outputm, xtm, tm)
        meanf, sigmaf = self.q_posterior_distribution_from_output_and_xt(outputf, xtf, tf)
        noisem = torch.randn_like(meanm)
        x_outm = meanm + sigmam * noisem
        noisef = torch.randn_like(meanf)
        x_outf = meanf + sigmaf * noisef
        x_out = torch.cat([x_outm, x_outf], dim=2)
        xstart = output
        return x_out, xstart

    def stmc_forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = n_seq, n_frames, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_seq = torch.full((n_seq,), diffusion_step)
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample_stmc(xt, y, t_seq, t_bs)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def p_sample_stmc(self, xt, y, t_seq, t_bs):
        all_intervals = y["infos"]["all_intervals"]

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        x_lst = []
        for idx, intervals in enumerate(all_intervals):
            x_lst.extend([xt[idx, x.start : x.end] for x in intervals])

        lengths = [len(x) for x in x_lst]
        assert lengths == y["length"]

        xx = collate_tensor_with_padding(x_lst)
        output = self.denoiser(xx, y, t_bs)

        if guidance_weight != 1.0:
            output_cond = output

            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xx, y_uncond, t_bs)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        output_xt = 0 * xt
        combine_features_intervals(output, y["infos"], output_xt)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            output_xt, xt, t_seq
        )

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise

        xstart = output_xt
        return x_out, xstart

    def sinc_baseline(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, max(lengths), nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t_bs)

        # at the end recombine
        shape = n_seq, n_frames, nfeats
        output = torch.zeros(shape, device=device)

        xstart = combine_features_intervals(xstart, infos, output)

        if "lerp" in infos["baseline"] or "interp" in infos["baseline"]:
            # interpolate to smooth the results
            xstart = interpolate_intervals(xstart, infos)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart


# Modify the GaussianDiffusionForcing class to extend DiscreteDiffusion
class GaussianDiffusionForcing(DiscreteDiffusion):
    """
    Diffusion Forcing: a diffusion model that allows for different noise levels for different tokens.
    
    This enables:
    1. Independent denoising of different tokens
    2. Flexible context conditioning
    3. Advanced sampling strategies with history guidance
    """
    
    name = "gaussian_forcing"
    
    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
        lr: float = 2e-4,
        g_text: bool = False,
        uni: bool = False,
        loss_type: str = 'SNR_MSE',
        clip_noise: float = 20.0,
        use_causal_mask: bool = False,
        loss_weighting=None,
        sampling_timesteps=100,
        ddim_sampling_eta: float = 0.0,  # Add as configurable parameter
        context_indices=None,  # Add context indices as an initialization parameter
        default_scheduling_matrix="pyramid",  # Default scheduling matrix type  full_sequence
        default_uncertainty_scale=15.0,  # Default uncertainty scale
        token_conditioning: str = "independent",# "independent", "pyramid", "full_sequence"
        schedule_sampling_prob: float = 1,  # Probability of using schedule vs independent sampling
        **kwargs
    ):

        """
        Initialize the Diffusion Forcing model.
        
        Args:
            denoiser: Model that predicts x₀ or noise.
            schedule: Beta schedule for the diffusion process.
            timesteps: Number of diffusion timesteps.
            motion_normalizer: Normalizer for motion features.
            text_normalizer: Normalizer for text features.
            prediction: Type of prediction ("x" or "eps").
            lr: Learning rate.
            g_text: Whether to use text guidance.
            uni: Whether to use unified loss.
            loss_type: Type of loss function.
            clip_noise: Maximum magnitude for noise.
            token_conditioning: How to condition on tokens ("independent", "uniform").
            use_causal_mask: Whether to use causal masking for autoregressive generation.
            context_indices: Default indices to use as context frames (can be overridden during sampling)
            default_scheduling_matrix: Default scheduling matrix type to use
            default_uncertainty_scale: Default uncertainty scale for scheduling
        """

        # Create a config object that will be compatible with DiscreteDiffusion
        cfg = DictConfig({
            "timesteps": timesteps,
            "sampling_timesteps": sampling_timesteps if sampling_timesteps is not None else timesteps,
            "beta_schedule": "cosine" if not hasattr(schedule, "name") else schedule.name,
            "schedule_fn_kwargs": {},
            "prediction": prediction,
            "loss_weighting": loss_weighting if loss_weighting is not None else {
                "strategy": "uniform",
                "sigmoid_bias": 5.0,
            },
            "ddim_sampling_eta": ddim_sampling_eta,  # Use the passed parameter
            "clip_noise": clip_noise,
            "use_causal_mask": use_causal_mask,
        })
        
        # Initialize DiscreteDiffusion with our parameters and model
        super().__init__(
            cfg=cfg,
            model=denoiser,
            x_shape=None,  # Use nfeats as shape if available
            custom_schedule=schedule if callable(schedule) else None
        )
        
        # Store additional attributes specific to GaussianDiffusionForcing
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer
        self.g_text = g_text
        self.uni = uni
        self.loss_type = loss_type
        self.token_conditioning = token_conditioning
        self.lr = lr
        self.schedule_sampling_prob = schedule_sampling_prob

        # Store context configuration
        self.context_indices = context_indices
        self.default_scheduling_matrix = default_scheduling_matrix
        self.default_uncertainty_scale = default_uncertainty_scale
        
        
        # Set up appropriate loss function
        if self.loss_type == 'MSE':
            self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")
        elif self.loss_type == 'WeightedMSE':
            self.reconstruction_loss = WeightedMSELoss(reduction="mean")
        elif self.loss_type == 'LearnableMSE':
            self.reconstruction_loss = LearnableMSELoss(initial_lambda=1.0, learn_lambda=True)
        elif self.loss_type == 'LearnableWeightedMSE':
            self.reconstruction_loss = LearnableWeightedMSELoss(initial_lambda=1.0, learn_lambda=True)
        elif self.loss_type == 'SNR_MSE':
            print("Using SNR-based loss for diffusion forcing")
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented")
    
    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def training_step(self, batch, batch_idx):
        bs = len(batch["x"])
        if self.uni:
            loss = self.diffusion_step_uni(batch, batch_idx, training=True)
        else:
            loss = self.diffusion_step(batch, batch_idx, training=True)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
            
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        bs = len(batch["x"])
        if self.uni:
            loss = self.diffusion_step_uni(batch, batch_idx, training=True)
        else:
            loss = self.diffusion_step(batch, batch_idx, training=True)
        # loss = self.diffusion_step(batch, batch_idx)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )

        return loss["loss"]

    def on_train_epoch_end(self):
        dico = {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.global_step),
        }
        # reset losses
        self._saved_losses = defaultdict(list)
        self.losses = []
        self.log_dict(dico)

    # Keep the existing methods from GaussianDiffusionForcing
    def prepare_tx_emb(self, tx_emb):
        # Text embedding normalization
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        tx = {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"].bool()),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }
        return tx

    def _reweight_loss(self, loss, weight=None):
        """
        Reweights and reduces loss based on optional weight mask.
        
        Args:
            loss: Loss tensor to reweight
            weight: Optional weight mask (e.g., attention mask)
            
        Returns:
            Scalar loss value
        """
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight
        
        return loss.mean()

    def diffusion_step(self, batch, batch_idx, training=False):
        """
        Perform a diffusion step with token-wise independent noise levels,
        leveraging DiscreteDiffusion functionality.
        
        Args:
            batch: Batch of data containing motion and text information
            batch_idx: Index of the batch
            training: Whether in training mode
            
        Returns:
            Dict containing loss information
        """
        mask = batch["mask"]
        motion_dim = batch["motion_dim"]
        text_dim = batch["text_dim"]        

        # Apply normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)
        if "stats_mask" in batch.keys():
            x = x * batch["stats_mask"]

        # Prepare conditioning
        if self.g_text:
            y = {
                "length": batch["length"],
                "mask": mask,
                "tx": self.prepare_tx_emb(batch["tx"]),
            }
        else:
            y = {
                "length": batch["length"],
                "mask": mask,
            }

        bs, n_tokens, *_ = x.shape
        
        # # Sample diffusion noise levels
        # # The key difference: sample noise levels differently for each token
        # if self.token_conditioning == "independent":
        #     # Independent noise levels for each token
        #     t = torch.randint(0, self.timesteps, (bs, n_tokens), device=x.device)
        # else:
        #     # Uniform noise levels (standard diffusion)
        #     t = torch.randint(0, self.timesteps, (bs,), device=x.device)
        #     t = repeat(t, "b -> b t", t=n_tokens)
        
        # Sample diffusion noise levels - UNIFIED LOGIC FOR ALL TYPES
        if self.token_conditioning == "independent":
            # Independent noise levels for each token
            t = torch.randint(0, self.timesteps, (bs, n_tokens), device=x.device)
        elif self.token_conditioning in ["pyramid", "autoregressive", "trapezoid", "full_sequence"]:
            # Use scheduling matrix sampling with probability
            if training and torch.rand(1).item() < self.schedule_sampling_prob:
                t = self._sample_timesteps_from_schedule(x.device, mask)
            else:
                # Fallback to independent sampling
                t = torch.randint(0, self.timesteps, (bs, n_tokens), device=x.device)
        else:
            raise ValueError(f"Unknown token_conditioning: {self.token_conditioning}")
                
                    
        # Ensure noise levels are masked properly
        t = torch.where(
            mask.bool(),
            t,
            torch.full_like(t, self.timesteps - 1)
        )
        
        # Create noise and apply it - use DiscreteDiffusion's q_sample method
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        
        # Use parent class's q_sample method
        xt = self.q_sample(x_start=x, k=t, noise=noise)
        xt = masked(xt, mask)
        
        # Handle inpainting if needed
        if "inpainting_mask" in batch.keys():
            self.inpainting_mask = batch["inpainting_mask"].unsqueeze(0)
            xt = self.inpainting_mask * xt + (1 - self.inpainting_mask) * x

        # Denoise with the model
        model_pred = self.model_predictions(
            x=xt, k=t, external_cond=y
        )
        pred = model_pred.model_out
        xstart  = model_pred.pred_x_start
        xstart = masked(xstart, mask)
        pred = masked(pred, mask)

        if self.prediction == "noise":
            target = noise
        elif self.prediction == "x":
            target = x
        elif self.prediction == "v":
            target = self.predict_v(x, t, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        target = masked(target.to(dtype=torch.float32), mask)

        if "inpainting_mask" in batch.keys():
            target = target*self.inpainting_mask
            pred = pred*self.inpainting_mask             

        # Compute loss based on loss type
        if self.loss_type == "LearnableMSE" or self.loss_type == "LearnableWeightedMSE":
            # Set up parameters for learnable losses
            args = [
                target[:,:,:motion_dim], 
                pred[:,:,:motion_dim], 
                target[:,:,-text_dim:], 
                pred[:,:,-text_dim:]
            ]
            
            xloss, loss_m, loss_f, lambda_val = self.reconstruction_loss(*args)
            
            loss = {
                "loss": xloss,
                "motion loss": loss_m,
                "text loss": loss_f,
                "lambda": lambda_val,
            }
        elif self.loss_type == "SNR_MSE":         
            # Calculate the MSE loss
            loss_per_token = F.mse_loss(target, pred, reduction='none')
            
            # Apply SNR-weighted loss - use parent class method
            loss_weights = self.compute_loss_weights(t, self.loss_weighting.strategy)
            loss_weights = self.add_shape_channels(loss_weights)
            loss_per_token = loss_per_token * loss_weights
            
            # Use _reweight_loss to apply mask and reduce
            if "mask" in batch:
                xloss = self._reweight_loss(loss_per_token, batch["mask"])
            else:
                xloss = self._reweight_loss(loss_per_token)
            # # Compute final loss with mask
            # if "mask" in batch:
            #     loss_mask = batch["mask"]
            #     loss_mask_expanded = loss_mask.unsqueeze(-1) if loss_mask.ndim < loss_per_token.ndim else loss_mask
            #     loss_per_token = loss_per_token * loss_mask_expanded
            
            # # Reduce loss
            # xloss = loss_per_token.reshape(loss_per_token.shape[0], -1).mean(1).mean()

            loss = {"loss": xloss}
        else:
            # Standard MSE loss
            xloss = self.reconstruction_loss(target, pred)
            loss = {"loss": xloss}
            
        return loss

    def _generate_scheduling_matrix(self, horizon: int, infos=None):
        """
        Generate noise level scheduling matrix for sampling.
        
        Args:
            horizon: Length of the sequence
            infos: Optional dictionary with additional parameters
        
        Returns:
            Scheduling matrix of shape [timesteps+1, horizon]
        """
        # Get scheduling type and parameters from infos or defaults
        if infos is None:
            infos = {}
        
        scheduling_type = infos.get("scheduling_matrix", getattr(self, "default_scheduling_matrix", "full_sequence"))
        uncertainty_scale = infos.get("uncertainty_scale", getattr(self, "default_uncertainty_scale", 1.0))
        
        
        # Ensure horizon is a CPU value
        if isinstance(horizon, torch.Tensor):
            horizon = horizon.cpu().item()
        
        # Create scheduling matrix based on the type
        if scheduling_type == "pyramid":
            return self._generate_pyramid_scheduling_matrix(horizon, uncertainty_scale)
        elif scheduling_type == "full_sequence":
            return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
        elif scheduling_type == "autoregressive":
            return self._generate_pyramid_scheduling_matrix(horizon, uncertainty_scale)
        elif scheduling_type == "trapezoid":
            return self._generate_trapezoid_scheduling_matrix(horizon, uncertainty_scale)
        else:
            # Default to full sequence
            return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        sampling_timesteps = self.sampling_timesteps
        height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
    
    def _sample_timesteps_from_schedule(self, device: torch.device, mask: torch.Tensor) -> torch.Tensor:
        """Sample timesteps using scheduling matrix approach."""
        bs, n_tokens = mask.shape
        actual_lengths = mask.sum(dim=1)
        max_actual_length = actual_lengths.max().item()
        
        if max_actual_length == 0:
            return torch.zeros((bs, n_tokens), dtype=torch.long, device=device)
        
        # Generate scheduling matrix
        if self.token_conditioning == "pyramid":
            scheduling_matrix = self._generate_pyramid_scheduling_matrix(max_actual_length, self.default_uncertainty_scale)
        elif self.token_conditioning == "autoregressive":
            scheduling_matrix = self._generate_pyramid_scheduling_matrix(max_actual_length, self.default_uncertainty_scale)
        elif self.token_conditioning == "trapezoid":
            scheduling_matrix = self._generate_trapezoid_scheduling_matrix(max_actual_length, self.default_uncertainty_scale)
        elif self.token_conditioning == "full_sequence":
            scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(max_actual_length, axis=1)
        else:
            raise ValueError(f"Unknown token_conditioning: {self.token_conditioning}")
        
        scheduling_matrix = torch.from_numpy(scheduling_matrix).long().to(device)
        
        # Calculate valid sampling ranges
        if self.token_conditioning == "pyramid":
            max_valid_rows = (self.sampling_timesteps + ((actual_lengths - 1) * self.default_uncertainty_scale).long() + 1)
        elif self.token_conditioning == "autoregressive":
            max_valid_rows = (self.sampling_timesteps + ((actual_lengths - 1) * self.default_uncertainty_scale).long() + 1)
        elif self.token_conditioning == "trapezoid":
            max_valid_rows = (self.sampling_timesteps + (((actual_lengths + 1) // 2) * self.default_uncertainty_scale).long())
        else:
            max_valid_rows = torch.full((bs,), scheduling_matrix.shape[0], dtype=torch.long, device=device)
        
        max_valid_rows = torch.clamp(max_valid_rows, 1, scheduling_matrix.shape[0])
        
        # Sample rows
        selected_rows = torch.zeros(bs, dtype=torch.long, device=device)
        for b in range(bs):
            if max_valid_rows[b] > 0:
                selected_rows[b] = torch.randint(0, max_valid_rows[b].item(), (1,), device=device)
        
        t = scheduling_matrix[selected_rows]
        real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=device).long()
        # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        t_real = real_steps[t]
        t_real = torch.clamp(t_real, 0, self.timesteps - 1)
        return t_real

    
    def batch_forward(
        self,
        batch,
        infos=None,
        progress_bar=tqdm,
    ):
        """
        Performs forward diffusion using data directly from a dataloader batch.
        Uses progressive chunk-based generation with sliding window.
        
        Args:
            batch: A batch from the dataloader containing motion and text data
            infos: Additional information dictionary (optional)
            progress_bar: Progress bar function (default: tqdm)
            
        Returns:
            Generated motion sequence
        """
        device = self.device
        
        # Extract information from batch
        mask = batch["mask"]
        motion_dim = batch["motion_dim"]
        text_dim = batch["text_dim"]   

        original_x = masked(self.motion_normalizer(batch["x"]), mask)
        if "stats_mask" in batch.keys():
            original_x = original_x * batch["stats_mask"] 
               
        # Initialize infos if not provided
        if infos is None:
            infos = {}
        
        # Prepare conditioning
        if self.g_text:
            conditions = {
                "length": batch["length"],
                "mask": mask,
                "tx": self.prepare_tx_emb(batch["tx"]),
                "infos": infos,
            }
            
            # Add unconditional embedding if guidance is needed
            guidance_weight = infos.get("guidance_weight", 1.0)
            if guidance_weight > 1.0 and "tx_uncond_batch" in batch:
                conditions["tx_uncond"] = self.prepare_tx_emb(batch["tx_uncond_batch"])
        else:
            conditions = {
                "length": batch["length"],
                "mask": mask,
                "infos": infos,
            }

        # Get dimensions
        batch_size = len(batch["length"])
        n_frames = max(batch["length"])
        if isinstance(n_frames, torch.Tensor):
            n_frames = n_frames.cpu().item()
        n_frames = int(n_frames)
        
        # Initialize for progressive generation
        xs_pred = []
        curr_frame = 0
        # xstart=[]
        
        # Handle context frames if provided
        context_frames = infos.get("context_frames", None)
        context_indices = infos.get("context_indices", getattr(self, "context_indices", None))
        n_context_frames = len(context_indices) if context_indices else 0
        
        if n_context_frames > 0 and context_frames is not None:
            # Initialize with context frames
            context_tensor = self.motion_normalizer(context_frames)
            context_tensor = masked(context_tensor, mask[:, :context_tensor.shape[1]])
            xs_pred = context_tensor
            curr_frame = n_context_frames
        
        # Set up chunking parameters
        chunk_size = infos.get("chunk_size", n_frames) 
        # chunk_size = 20
        n_tokens = infos.get("n_tokens", n_frames)  # Max tokens to process at once
        
        # Create progress bar
        pbar = progress_bar(total=n_frames, initial=curr_frame, desc="Sampling") if progress_bar else None
        
        # Progressive generation loop
        while curr_frame < n_frames:
            # Determine horizon for current chunk
            if chunk_size > 0:
                horizon = min(n_frames - curr_frame, chunk_size)
            else:
                horizon = n_frames - curr_frame
            
            # Generate scheduling matrix for this chunk
            scheduling_matrix = self._generate_scheduling_matrix(horizon, infos)
            
            # Generate noise for new chunk
            chunk_shape = (batch_size, horizon, motion_dim+text_dim)
            chunk = torch.randn(chunk_shape, device=device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            mask = mask[:, curr_frame:horizon]
            chunk = masked(chunk, mask)
            
            # Concatenate with existing prediction
            if len(xs_pred) == 0:
                xs_pred = chunk
            else:
                xs_pred = torch.cat([xs_pred, chunk], dim=1)
            
            # Determine sliding window start frame (for efficiency)
            start_frame = max(0, curr_frame + horizon - n_tokens)
            
            # Update progress bar if provided
            if pbar:
                pbar.set_postfix({
                    "start": start_frame,
                    "end": curr_frame + horizon,
                })
            
            # Generate through noise levels
            # for m in range(scheduling_matrix.shape[0] - 1):
            for m in range(100):
                # When creating the noise levels:
                from_noise_levels = np.zeros((batch_size, n_frames), dtype=np.int64)
                to_noise_levels = np.zeros((batch_size, n_frames), dtype=np.int64)

                # Set noise levels for the current chunk - note the axis changes
                from_noise_levels[:, curr_frame:curr_frame+horizon] = scheduling_matrix[m][None, :].repeat(batch_size, axis=0)
                to_noise_levels[:, curr_frame:curr_frame+horizon] = scheduling_matrix[m+1][None, :].repeat(batch_size, axis=0)
                
                # Convert to tensors
                from_noise_levels = torch.from_numpy(from_noise_levels).to(device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(device)
                
                from_noise_levels = torch.where(
                    mask.bool(),
                    from_noise_levels,
                    torch.full_like(from_noise_levels, self.sampling_timesteps)
                )
                to_noise_levels = torch.where(
                    mask.bool(),
                    to_noise_levels,
                    torch.full_like(to_noise_levels, self.sampling_timesteps)
                )
                # Handle inpainting if needed
                if "inpainting_mask" in batch:
                    self.inpainting_mask = batch["inpainting_mask"]
                    xs_pred = self.inpainting_mask * xs_pred + (1 - self.inpainting_mask) * original_x
                    # xs_pred = torch.where(
                    #     self.inpainting_mask.unsqueeze(0),
                    #     xs_pred,
                    #     original_x
                    # )
                
                # # Update xs_pred by DDIM or DDPM sampling
                # # Only update the frames in the sliding window
                # xs_pred[:, start_frame:], xstart = self.sample_step(
                #     x=xs_pred[:, start_frame:],
                #     curr_noise_level=from_noise_levels[:, start_frame:],
                #     next_noise_level=to_noise_levels[:, start_frame:],
                #     external_cond=conditions
                # )
                
                ###this section is just for debug####
                real_steps = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=xs_pred.device).long()
                # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
                from_noise_levels = real_steps[from_noise_levels]
                from_noise_levels = torch.clamp(from_noise_levels, 0, self.timesteps - 1)
                noise = torch.randn_like(original_x)
                noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
                xt = masked(noise, mask)
                from_noise_levels = torch.where(
                    mask.bool(),
                    from_noise_levels,
                    torch.full_like(from_noise_levels, self.timesteps - 1)
                )
                xs_pred[:, start_frame:], xstart = self.p_sample(
                    xt=xt[:, start_frame:],
                    t=from_noise_levels[:, start_frame:],
                    y=conditions,
                    x=original_x,
                    mask=mask
                )

                # Apply mask to respect sequence length
                xs_pred = masked(xs_pred, mask)
                xstart = masked(xstart, mask)
                xloss = self.reconstruction_loss(xstart*self.inpainting_mask, original_x*self.inpainting_mask)
                print(xloss)
            # Update position and progress bar
            curr_frame += horizon
            if pbar:
                pbar.update(horizon)
        
        
        
        
        # Denormalize the output
        result = self.motion_normalizer.inverse(xstart)
        return result

    def p_sample(self, xt, y, t, x, mask):
        # guided forward
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        xt = self.q_sample(x_start=x, k=t, noise=noise)
        xt = masked(xt, mask)
        print(t[0])
        # Handle inpainting if needed
        xt = self.inpainting_mask * xt + (1 - self.inpainting_mask) * x
        model_pred = self.model_predictions(
            x=xt,
            k=t,
            external_cond=y,
            external_cond_mask=None,
        )
        xstart = model_pred.pred_x_start
        x_out = model_pred.model_out
        return x_out, xstart

# class GaussianSRM(GaussianDiffusionForcing):
#     """
#     Gaussian Diffusion with Spatio-temporal Reasoning via advanced time sampling (SRM).
    
#     Based on GaussianDiffusionForcing but uses the advanced time sampler that allows
#     different timesteps for different tokens in the sequence.
#     """
    
#     name = "gaussian_srm"
    
#     def __init__(
#         self,
#         denoiser,
#         schedule,
#         timesteps,
#         motion_normalizer,
#         text_normalizer,
#         prediction: str = "x",
#         lr: float = 2e-4,
#         g_text: bool = False,
#         uni: bool = False,
#         loss_type: str = 'SNR_MSE',
#         clip_noise: float = 20.0,
#         # token_conditioning: str = "independent",
#         use_causal_mask: bool = False,
#         loss_weighting=None,
#         sampling_timesteps=None,
#         ddim_sampling_eta: float = 0.0,
#         # New parameters for time sampler
#         time_sampler_cfg: TimeSamplerCfg = None,
#         max_sequence_length: int = 512,
#         **kwargs
#     ):
#         # Initialize parent class
#         super().__init__(
#             denoiser=denoiser,
#             schedule=schedule,
#             timesteps=timesteps,
#             motion_normalizer=motion_normalizer,
#             text_normalizer=text_normalizer,
#             prediction=prediction,
#             lr=lr,
#             g_text=g_text,
#             uni=uni,
#             loss_type=loss_type,
#             clip_noise=clip_noise,
#             # token_conditioning=token_conditioning,
#             use_causal_mask=use_causal_mask,
#             loss_weighting=loss_weighting,
#             sampling_timesteps=sampling_timesteps,
#             ddim_sampling_eta=ddim_sampling_eta,
#             **kwargs
#         )
        
#         # Initialize advanced time sampler
#         if time_sampler_cfg is None:
#             # Default to independent sampling for each token
#             time_sampler_cfg = IndependentCfg(name="independent")
            
#         # Adapt the time sampler for 1D motion sequences instead of 2D patches
#         # The "resolution" concept maps to sequence length for motion
#         self.motion_time_sampler = get_time_sampler(
#             time_sampler_cfg, 
#             resolution=(max_sequence_length, 1)  # 1D sequence
#         )
#         self.max_sequence_length = max_sequence_length

#     def diffusion_step(self, batch, batch_idx, training=False):
#         """
#         Enhanced diffusion step using advanced time sampling.
#         Main difference: uses motion_time_sampler instead of simple random sampling.
#         """
#         mask = batch["mask"]
#         motion_dim = batch["motion_dim"]
#         text_dim = batch["text_dim"]        

#         # Apply normalization (same as parent)
#         x = masked(self.motion_normalizer(batch["x"]), mask)
#         if "stats_mask" in batch.keys():
#             x = x * batch["stats_mask"]

#         # Prepare conditioning (same as parent)
#         if self.g_text:
#             y = {
#                 "length": batch["length"],
#                 "mask": mask,
#                 "tx": self.prepare_tx_emb(batch["tx"]),
#             }
#         else:
#             y = {
#                 "length": batch["length"],
#                 "mask": mask,
#             }

#         bs, n_tokens, *_ = x.shape
        
#         sequence_length = min(n_tokens, self.max_sequence_length)
        
#         t_continuous, loss_weight = self.motion_time_sampler(
#             batch_size=bs, 
#             num_samples=1, 
#             device=x.device
#         )
        
#         # Convert from continuous [0,1] to discrete timesteps and reshape
#         t_continuous = t_continuous.squeeze(1)  # Remove samples dimension
#         loss_weight = loss_weight.squeeze(1)
        
#         # Take only the sequence length we need
#         t_continuous = t_continuous[:, :sequence_length]
#         loss_weight = loss_weight[:, :sequence_length]
        
#         # Pad or truncate to match actual sequence length
#         if sequence_length < n_tokens:
#             # Pad with uniform random values for longer sequences
#             extra_t = torch.rand(bs, n_tokens - sequence_length, device=x.device)
#             extra_weights = torch.ones(bs, n_tokens - sequence_length, device=x.device)
#             t_continuous = torch.cat([t_continuous, extra_t], dim=1)
#             loss_weight = torch.cat([loss_weight, extra_weights], dim=1)
#         elif sequence_length > n_tokens:
#             # Truncate for shorter sequences
#             t_continuous = t_continuous[:, :n_tokens]
#             loss_weight = loss_weight[:, :n_tokens]
        
#         # Convert to discrete timesteps
#         t = (t_continuous * (self.timesteps - 1)).long()
        
#         # Rest is the same as parent class...
        
#         # Ensure noise levels are masked properly
#         t = torch.where(mask.bool(), t, torch.full_like(t, self.timesteps - 1))
#         loss_weight = torch.where(mask.bool(), loss_weight, torch.zeros_like(loss_weight))
        
#         # Create noise and apply it - use DiscreteDiffusion's q_sample method
#         noise = torch.randn_like(x)
#         noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        
#         # Use parent class's q_sample method
#         xt = self.q_sample(x_start=x, k=t, noise=noise)
#         xt = masked(xt, mask)
        
#         # Handle inpainting if needed
#         if "inpainting_mask" in batch.keys():
#             self.inpainting_mask = batch["inpainting_mask"].unsqueeze(0)
#             xt = self.inpainting_mask * xt + (1 - self.inpainting_mask) * x

#         # Denoise with the model
#         model_pred = self.model_predictions(x=xt, k=t, external_cond=y)
#         pred = model_pred.model_out
#         xstart = model_pred.pred_x_start
#         xstart = masked(xstart, mask)
#         pred = masked(pred, mask)

#         if self.prediction == "noise":
#             target = noise
#         elif self.prediction == "x":
#             target = x
#         elif self.prediction == "v":
#             target = self.predict_v(x, t, noise)
#         else:
#             raise ValueError(f"unknown objective {self.prediction}")

#         target = masked(target.to(dtype=torch.float32), mask)

#         if "inpainting_mask" in batch.keys():
#             target = target*self.inpainting_mask
#             pred = pred*self.inpainting_mask             

#         # Compute loss based on loss type
#         if self.loss_type == "LearnableMSE" or self.loss_type == "LearnableWeightedMSE":
#             args = [
#                 target[:,:,:motion_dim], 
#                 pred[:,:,:motion_dim], 
#                 target[:,:,-text_dim:], 
#                 pred[:,:,-text_dim:]
#             ]
            
#             xloss, loss_m, loss_f, lambda_val = self.reconstruction_loss(*args)
            
#             loss = {
#                 "loss": xloss,
#                 "motion loss": loss_m,
#                 "text loss": loss_f,
#                 "lambda": lambda_val,
#             }
#         elif self.loss_type == "SNR_MSE":         
#             # Calculate the MSE loss
#             loss_per_token = F.mse_loss(target, pred, reduction='none')
            
#             # Apply SNR-weighted loss - use parent class method
#             snr_weights = self.compute_loss_weights(t, self.loss_weighting.strategy)
#             snr_weights = self.add_shape_channels(snr_weights)
            
#             # KEY DIFFERENCE: Combine SNR weights with importance sampling weights
#             combined_weights = snr_weights * loss_weight.unsqueeze(-1)
#             loss_per_token = loss_per_token * combined_weights
            
#             # Use _reweight_loss to apply mask and reduce
#             if "mask" in batch:
#                 xloss = self._reweight_loss(loss_per_token, batch["mask"])
#             else:
#                 xloss = self._reweight_loss(loss_per_token)

#             loss = {"loss": xloss}
#         else:
#             # Standard MSE loss with importance weighting
#             mse_loss = self.reconstruction_loss(target, pred)
#             # Apply importance weights
#             weighted_loss = mse_loss * loss_weight.mean()  # Average importance weight
#             loss = {"loss": weighted_loss}
            
#         return loss
