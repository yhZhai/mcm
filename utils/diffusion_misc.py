import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from utils.matching import global_correlation_softmax, local_correlation_softmax


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(
    model_output, timesteps, sample, prediction_type, alphas, sigmas
):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        self.device = None
        self._reset(alpha_cumprods, timesteps, ddim_timesteps)

    def _reset(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

        if self.device is not None:
            self.to(self.device)

    def to(self, device):
        self.device = device
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_emebddings = text_encoder(text_input_ids.to(text_encoder.device))
        prompt_embeds = prompt_emebddings[0]
        pooler_output = prompt_emebddings[1]

    return {
        "prompt_embeds": prompt_embeds,
        "pooler_output": pooler_output,
    }


def get_module_kohya_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(
                module.peft_config[adapter_name].lora_alpha
            ).to(dtype)

    return kohya_ss_state_dict


# from freeinit
def get_free_init_freq_filter(
    shape: Tuple[int, ...],
    device: Union[str, torch.dtype],
    filter_type: str = "butterworth",
    order: float = 4,
    spatial_stop_frequency: float = 0.25,
    temporal_stop_frequency: float = 0.25,
) -> torch.Tensor:
    r"""Returns the FreeInit filter based on filter type and other input conditions."""

    time, height, width = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)

    if spatial_stop_frequency == 0 or temporal_stop_frequency == 0:
        return mask

    if filter_type == "butterworth":

        def retrieve_mask(x):
            return 1 / (1 + (x / spatial_stop_frequency**2) ** order)

    elif filter_type == "gaussian":

        def retrieve_mask(x):
            return math.exp(-1 / (2 * spatial_stop_frequency**2) * x)

    elif filter_type == "ideal":

        def retrieve_mask(x):
            return 1 if x <= spatial_stop_frequency * 2 else 0

    else:
        raise NotImplementedError(
            "`filter_type` must be one of gaussian, butterworth or ideal"
        )

    for t in range(time):
        for h in range(height):
            for w in range(width):
                d_square = (
                    (
                        (spatial_stop_frequency / temporal_stop_frequency)
                        * (2 * t / time - 1)
                    )
                    ** 2
                    + (2 * h / height - 1) ** 2
                    + (2 * w / width - 1) ** 2
                )
                mask[..., t, h, w] = retrieve_mask(d_square)

    return mask.to(device)


def apply_freq_filter(
    x: torch.Tensor, low_pass_filter: torch.Tensor, out_freq: str = "low"
) -> torch.Tensor:
    r"""Noise reinitialization."""
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))

    # IFFT
    if out_freq == "low":
        x_freq_low = x_freq * low_pass_filter
        x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
        x_low = fft.ifftn(x_freq_low, dim=(-3, -2, -1)).real
        return x_low
    elif out_freq == "high":
        high_pass_filter = 1 - low_pass_filter
        x_freq_high = x_freq * high_pass_filter
        x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
        x_high = fft.ifftn(x_freq_high, dim=(-3, -2, -1)).real
        return x_high
    else:
        raise ValueError(f"Invalid out_freq: {out_freq}")


def prepare_cd_target(latent, target: str, spatial_head: Optional[nn.Module] = None):
    # latent shape: b, c, t, h, w
    b, c, t, h, w = latent.shape

    if target in ["raw", "learn", "hlearn"]:
        return latent
    # elif target == "learn":
    #     latent = spatial_head(latent)
    #     return latent
    elif target == "diff":
        latent_prev = latent[:, :, :-1]
        latent_next = latent[:, :, 1:]
        diff = latent_next - latent_prev
        return diff
    elif target in ["freql", "freqh"]:
        shape = latent.shape
        shape = (1, *shape[1:])
        low_pass_filter = get_free_init_freq_filter(shape, latent.device)
        if target == "freql":
            return apply_freq_filter(latent, low_pass_filter, out_freq="low")
        elif target == "freqh":
            return apply_freq_filter(latent, low_pass_filter, out_freq="high")
        else:
            raise ValueError(f"Invalid target: {target}")
    elif target in ["lcor", "gcor", "sgcor", "sgcord"]:
        latent_prev = latent[:, :, :-1]
        latent_next = latent[:, :, 1:]
        latent_prev = rearrange(latent_prev, "b c t h w -> (b t) c h w")
        latent_next = rearrange(latent_next, "b c t h w -> (b t) c h w")

        if target == "lcor":
            flow, _, corr = local_correlation_softmax(latent_prev, latent_next, 7)
            return corr
        elif target in ["gcor", "sgcor", "sgcord"]:
            flow, _, corr = global_correlation_softmax(latent_prev, latent_next)
            if target == "gcor":
                return corr
            elif target == "sgcor":
                return corr * (h**0.5)
            elif target == "sgcord":
                raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError(f"Invalid target: {target}")
