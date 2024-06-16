#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import functools
import gc
import logging
import math
import os
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import accelerate
import diffusers
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AnimateDiffPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    MotionAdapter,
    StableDiffusionPipeline,
    TextToVideoSDPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
    UNetMotionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.animatediff.pipeline_animatediff import tensor2vid
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from tabulate import tabulate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize, RandomCrop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig

from args import parse_args
from dataset.webvid_dataset_wbd import Text2VideoDataset
from models.discriminator_handcraft import (
    ProjectedDiscriminator,
    get_dino_features,
    preprocess_dino_input,
)
from models.spatial_head import IdentitySpatialHead, SpatialHead
from utils.diffusion_misc import *
from utils.dist import dist_init, dist_init_wo_accelerate, get_deepspeed_config
from utils.misc import *
from utils.wandb import setup_wandb

MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(name)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def save_to_local(save_dir: str, prompt: str, video):
    if len(prompt) > 256:
        prompt = prompt[:256]
    prompt = prompt.replace(" ", "_")
    logger.info(f"Saving images to {save_dir}")

    export_to_video(video, os.path.join(save_dir, f"{prompt}.mp4"))


def log_validation(
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    name="target",
    scheduler: str = "lcm",
    num_inference_steps: int = 4,
    add_to_trackers: bool = True,
    use_lora: bool = False,
    disc_gt_images: Optional[List] = None,
    guidance_scale: float = 1.0,
    spatial_head: Optional = None,
    logger_prefix: str = "",
):
    logger.info("Running validation... ")
    scheduler_additional_kwargs = {}
    if args.base_model_name == "animatediff":
        scheduler_additional_kwargs["beta_schedule"] = "linear"
        scheduler_additional_kwargs["clip_sample"] = False
        scheduler_additional_kwargs["timestep_spacing"] = "linspace"

    if scheduler == "lcm":
        # set beta_schedule="linear" according to https://huggingface.co/wangfuyun/AnimateLCM
        scheduler = LCMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    elif scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    elif scheduler == "euler":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            **scheduler_additional_kwargs,
        )
    else:
        raise ValueError(f"Scheduler {scheduler} is not supported.")

    unet = deepcopy(accelerator.unwrap_model(unet))
    if args.base_model_name == "animatediff":
        pipeline_cls = AnimateDiffPipeline
    elif args.base_model_name == "modelscope":
        pipeline_cls = TextToVideoSDPipeline

    if use_lora:
        pipeline = pipeline_cls.from_pretrained(
            args.pretrained_teacher_model,
            vae=vae,
            scheduler=scheduler,
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )
        lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
        pipeline.load_lora_weights(lora_state_dict)
        pipeline.fuse_lora()
    else:
        pipeline = pipeline_cls.from_pretrained(
            args.pretrained_teacher_model,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            revision=args.revision,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)

    if (
        args.enable_xformers_memory_efficient_attention
        and args.base_model_name != "animatediff"
    ):
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "Cute small corgi sitting in a movie theater eating popcorn, unreal engine.",
        "A Pikachu with an angry expression and red eyes, with lightning around it, hyper realistic style.",
        "A dog is reading a thick book.",
        "Three cats having dinner at a table at new years eve, cinematic shot, 8k.",
        "An astronaut riding a pig, highly realistic dslr photo, cinematic shot.",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        output = []
        with torch.autocast("cuda", dtype=weight_dtype):
            output = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
                guidance_scale=guidance_scale,
                output_type="latent",
            ).frames
            if spatial_head is not None:
                output = spatial_head(output)

            output = pipeline.decode_latents(output)
            video = tensor2vid(output, pipeline.image_processor, output_type="np")
            # video should be a tensor of shape (t, h, w, 3), min 0, max 1
            video = video[0]

        save_dir = os.path.join(args.output_dir, "output", f"{name}-step-{step}")
        if accelerator.is_main_process:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        image_logs.append({"validation_prompt": prompt, "video": video})
        save_to_local(save_dir, prompt, video)

    if add_to_trackers:
        try:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    for log in image_logs:
                        images = log["video"]
                        validation_prompt = (
                            f"{logger_prefix}{num_inference_steps} steps/"
                            + log["validation_prompt"]
                        )
                        formatted_images = []
                        for image in images:
                            formatted_images.append(np.asarray(image))

                        formatted_images = np.stack(formatted_images)

                        tracker.writer.add_images(
                            validation_prompt,
                            formatted_images,
                            step,
                            dataformats="NHWC",
                        )
                    if disc_gt_images is not None:
                        for i, image in enumerate(disc_gt_images):
                            tracker.writer.add_image(
                                f"discriminator gt image/{i}",
                                image,
                                step,
                                dataformats="HWC",
                            )
                elif tracker.name == "wandb":
                    # log image for comparison
                    formatted_images = []

                    for log in image_logs:
                        images = log["video"]
                        validation_prompt = log["validation_prompt"]
                        image = wandb.Image(images[0], caption=validation_prompt)
                        formatted_images.append(image)

                    if args.use_lora:
                        tracker.log(
                            {
                                f"{logger_prefix}validation image {num_inference_steps} steps": formatted_images
                            },
                            step=step,
                        )
                    else:
                        tracker.log(
                            {
                                f"{logger_prefix}validation image {num_inference_steps} steps/{name}": formatted_images
                            },
                            step=step,
                        )

                    # log video
                    formatted_video = []
                    for log in image_logs:
                        video = (log["video"] * 255).astype(np.uint8)
                        validation_prompt = log[
                            "validation_prompt"
                        ]  # wandb does not support video logging with caption
                        video = wandb.Video(
                            np.transpose(video, (0, 3, 1, 2)), fps=4, format="mp4"
                        )
                        formatted_video.append(video)

                    if args.use_lora:
                        tracker.log(
                            {
                                f"{logger_prefix}validation video {num_inference_steps} steps": formatted_video
                            },
                            step=step,
                        )
                    else:
                        tracker.log(
                            {
                                f"{logger_prefix}validation video {num_inference_steps} steps/{name}": formatted_video
                            },
                            step=step,
                        )
                    # log discriminator ground truth images
                    if disc_gt_images is not None:
                        formatted_disc_gt_images = []
                        for i, image in enumerate(disc_gt_images):
                            image = wandb.Image(
                                image, caption=f"discriminator gt image {i}"
                            )
                            formatted_disc_gt_images.append(image)
                        tracker.log(
                            {"discriminator gt images": formatted_disc_gt_images},
                            step=step,
                        )
                else:
                    logger.warning(f"image logging not implemented for {tracker.name}")
        except Exception as e:
            logger.error(f"Failed to log images: {e}")

    del pipeline
    del unet
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def main(args):
    # torch.multiprocessing.set_sharing_strategy("file_system")
    dist_init()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    setup_wandb()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
        # deepspeed_plugin=deepspeed_plugin,
    )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Make one log on every process with the configuration for debugging.
    logger.info("Printing accelerate state", main_process_only=False)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_batch_size / 128
        args.disc_learning_rate = (
            args.disc_learning_rate * total_batch_size * args.disc_tsn_num_frames / 128
        )
        logger.info(f"Scaling learning rate to {args.learning_rate}")
        logger.info(f"Scaling discriminator learning rate to {args.disc_learning_rate}")

    sorted_args = sorted(vars(args).items())
    logger.info(
        "\n" + tabulate(sorted_args, headers=["key", "value"], tablefmt="rounded_grid"),
        main_process_only=True,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    try:
        accelerator.wait_for_everyone()
    except Exception as e:
        logger.error(f"Failed to wait for everyone: {e}")
        dist_init_wo_accelerate()
        accelerator.wait_for_everyone()

    # 1. Create the noise scheduler and the desired noise schedule.
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            revision=args.teacher_revision,
            rescale_betas_zero_snr=True if args.zero_snr else False,
            beta_schedule=args.beta_schedule,
        )
    except Exception as e:
        logger.error(f"Failed to load the noise scheduler: {e}")
        logger.info("Switching to online pretrained checkpoint")
        args.pretrained_teacher_model = args.online_pretrained_teacher_model
        args.motion_adapter_path = args.online_motion_adapter_path
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler",
            revision=args.teacher_revision,
            rescale_betas_zero_snr=True if args.zero_snr else False,
            beta_schedule=args.beta_schedule,
        )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # Initialize the DDIM ODE solver for distillation.
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizers from SD 1.X/2.X checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.teacher_revision,
        use_fast=False,
    )

    # 3. Load text encoders from SD 1.X/2.X checkpoint.
    # import correct text encoder classes
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.teacher_revision,
    )

    # 4. Load VAE from SD 1.X/2.X checkpoint
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.teacher_revision,
    )

    # 5. Load teacher U-Net from SD 1.X/2.X checkpoint
    if args.base_model_name == "animatediff":
        teacher_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="unet",
            revision=args.teacher_revision,
        )
        teacher_motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
        teacher_unet = UNetMotionModel.from_unet2d(teacher_unet, teacher_motion_adapter)
    elif args.base_model_name == "modelscope":
        teacher_unet = UNet3DConditionModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="unet",
            revision=args.teacher_revision,
        )

    # 5.1 Load DINO
    dino = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )
    ckpt_path = "weights/dinov2_vits14_pretrain.pth"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    dino.load_state_dict(state_dict)
    logger.info(f"Loaded DINO model from {ckpt_path}")
    dino.eval()

    # 5.2 Load sentence-level CLIP
    open_clip_model, *_ = open_clip.create_model_and_transforms(
        "ViT-g-14",
        pretrained="weights/open_clip_pytorch_model.bin",
    )
    open_clip_tokenizer = open_clip.get_tokenizer("ViT-g-14")

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    dino.requires_grad_(False)
    open_clip_model.requires_grad_(False)
    normalize_fn = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # 7. Create online student U-Net.
    # For whole model fine-tuning, this will be updated by the optimizer (e.g.,
    # via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    if args.use_lora:
        if args.base_model_name == "animatediff":
            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_teacher_model,
                subfolder="unet",
                revision=args.teacher_revision,
            )
            motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)
        elif args.base_model_name == "modelscope":
            unet = UNet3DConditionModel.from_pretrained(
                args.pretrained_teacher_model,
                subfolder="unet",
                revision=args.teacher_revision,
            )
    else:
        assert (
            args.base_model_name == "animatediff"
        ), f"Please use LoRA for {args.base_model_name}"

        time_cond_proj_dim = (
            teacher_unet.config.time_cond_proj_dim
            if "time_cond_proj_dim" in teacher_unet.config
            and teacher_unet.config.time_cond_proj_dim is not None
            else args.unet_time_cond_proj_dim
        )
        if args.base_model_name == "animatediff":
            unet = UNetMotionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )

            # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
            # Initialize from (online) unet
            target_unet = UNetMotionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )
        elif args.base_model_name == "modelscope":
            raise NotImplementedError

            unet = UNet3DConditionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )

            # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
            # Initialize from (online) unet
            target_unet = UNet3DConditionModel.from_config(
                teacher_unet.config, time_cond_proj_dim=time_cond_proj_dim
            )
        # load teacher_unet weights into unet
        unet.load_state_dict(teacher_unet.state_dict(), strict=False)
        target_unet.load_state_dict(unet.state_dict())
        target_unet.train()
        target_unet.requires_grad_(False)

        # freeze non-motion module parameters
        for param_name, param in unet.named_parameters():
            if "motion_modules" not in param_name.lower():
                param.requires_grad_(False)

        # count trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in unet.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    if args.cd_target in ["learn", "hlearn"]:
        if args.cd_target == "learn":
            spatial_head = SpatialHead(num_channels=4, num_layers=2, kernel_size=1)
            target_spatial_head = SpatialHead(
                num_channels=4, num_layers=2, kernel_size=1
            )
            logger.info("Using SpatialHead for spatial head")
        elif args.cd_target == "hlearn":
            spatial_head = SpatialHead(num_channels=4, num_layers=5, kernel_size=3)
            target_spatial_head = SpatialHead(
                num_channels=4, num_layers=5, kernel_size=3
            )
            logger.info("Using SpatialHead for spatial head")
        else:
            raise ValueError(f"cd_target {args.cd_target} is not supported.")

        spatial_head.train()
        target_spatial_head.load_state_dict(spatial_head.state_dict())
        target_spatial_head.train()
        target_spatial_head.requires_grad_(False)
    else:
        spatial_head = None
        target_spatial_head = None

    unet.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    if args.use_lora:
        if args.lora_target_modules is not None:
            logger.warning(
                "We are currently ignoring the `lora_target_modules` argument. As of now, LoRa does not support Conv3D layers."
            )
            lora_target_modules = [
                module_key.strip() for module_key in args.lora_target_modules.split(",")
            ]
        else:
            lora_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ]

        # Currently LoRA does not support Conv3D, thus removing the Conv3D
        # layers from the list of target modules.
        key_list = []
        for name, module in unet.named_modules():
            if any([name.endswith(module_key) for module_key in lora_target_modules]):
                if args.base_model_name == "modelscope" and not (
                    "temp" in name and "conv" in name
                ):
                    key_list.append(name)
                elif args.base_model_name == "animatediff":
                    key_list.append(name)

        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=key_list,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        unet = get_peft_model(unet, lora_config)

        if (
            args.from_pretrained_unet is not None
            and args.from_pretrained_unet != "None"
        ):
            # TODO currently only supports LoRA
            logger.info(f"Loading pretrained UNet from {args.from_pretrained_unet}")
            unet.load_adapter(
                args.from_pretrained_unet,
                "default",
                is_trainable=True,
                torch_device="cpu",
            )
        unet.print_trainable_parameters()

    # 8.1. Create discriminator for the student U-Net.
    c_dim = 1024
    discriminator = ProjectedDiscriminator(
        embed_dim=dino.embed_dim, c_dim=c_dim
    )  # TODO add dino name and patch size
    if args.from_pretrained_disc is not None and args.from_pretrained_disc != "None":
        try:
            disc_state_dict = load_file(
                os.path.join(
                    args.from_pretrained_disc,
                    "discriminator",
                    "diffusion_pytorch_model.safetensors",
                )
            )
            discriminator.load_state_dict(disc_state_dict)
            logger.info(
                f"Loaded pretrained discriminator from {args.from_pretrained_disc}"
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained discriminator: {e}")
    discriminator.train()

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    dino.to(accelerator.device, dtype=weight_dtype)
    open_clip_model.to(accelerator.device)

    # Move teacher_unet to device, optionally cast to weight_dtype
    if not args.use_lora:
        target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)
    if args.cd_target in ["learn", "hlearn"]:
        target_spatial_head.to(accelerator.device)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    # Move the ODE solver to accelerator.device.
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if args.use_lora:

            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    unet_ = accelerator.unwrap_model(unet)
                    lora_state_dict = get_peft_model_state_dict(
                        unet_, adapter_name="default"
                    )
                    StableDiffusionPipeline.save_lora_weights(
                        os.path.join(output_dir, "unet_lora"), lora_state_dict
                    )
                    # save weights in peft format to be able to load them back
                    unet_.save_pretrained(output_dir)

                    discriminator_ = accelerator.unwrap_model(discriminator)
                    discriminator_.save_pretrained(
                        os.path.join(output_dir, "discriminator")
                    )

                    if args.cd_target in ["learn", "hlearn"]:
                        spatial_head_ = accelerator.unwrap_model(spatial_head)
                        spatial_head_.save_pretrained(
                            os.path.join(output_dir, "spatial_head")
                        )
                        target_spatial_head_ = accelerator.unwrap_model(
                            target_spatial_head
                        )
                        target_spatial_head_.save_pretrained(
                            os.path.join(output_dir, "target_spatial_head")
                        )

                    for _, model in enumerate(models):
                        # make sure to pop weight so that corresponding model is not saved again
                        if len(weights) > 0:
                            weights.pop()

        else:
            # only support finetune motion module for AnimateDiff
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    target_unet_ = accelerator.unwrap_model(target_unet)
                    target_unet_.save_motion_modules(
                        os.path.join(output_dir, "target_motion_modules")
                    )

                    unet_ = accelerator.unwrap_model(unet)
                    unet_.save_motion_modules(
                        os.path.join(output_dir, "motion_modules")
                    )

                    discriminator_ = accelerator.unwrap_model(discriminator)
                    discriminator_.save_pretrained(
                        os.path.join(output_dir, "discriminator")
                    )

                    if args.cd_target in ["learn", "hlearn"]:
                        spatial_head_ = accelerator.unwrap_model(spatial_head)
                        spatial_head_.save_pretrained(
                            os.path.join(output_dir, "spatial_head")
                        )
                        target_spatial_head_ = accelerator.unwrap_model(
                            target_spatial_head
                        )
                        target_spatial_head_.save_pretrained(
                            os.path.join(output_dir, "target_spatial_head")
                        )

                    for i, model in enumerate(models):
                        # make sure to pop weight so that corresponding model is not saved again
                        if len(weights) > 0:
                            weights.pop()

        if args.use_lora:

            def load_model_hook(models, input_dir):
                # load the LoRA into the model
                unet_ = accelerator.unwrap_model(unet)
                unet_.load_adapter(
                    input_dir, "default", is_trainable=True, torch_device="cpu"
                )

                disc_state_dict = load_file(
                    os.path.join(
                        input_dir,
                        "discriminator",
                        "diffusion_pytorch_model.safetensors",
                    )
                )
                disc_ = accelerator.unwrap_model(discriminator)
                disc_.load_state_dict(disc_state_dict)
                del disc_state_dict

                if args.cd_target in ["learn", "hlearn"]:
                    spatial_head_state_dict = load_file(
                        os.path.join(
                            input_dir,
                            "spatial_head",
                            "diffusion_pytorch_model.safetensors",
                        )
                    )
                    spatial_head_ = accelerator.unwrap_model(spatial_head)
                    spatial_head_.load_state_dict(spatial_head_state_dict)
                    del spatial_head_state_dict
                    target_spatial_head_state_dict = load_file(
                        os.path.join(
                            input_dir,
                            "target_spatial_head",
                            "diffusion_pytorch_model.safetensors",
                        )
                    )
                    target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
                    target_spatial_head_.load_state_dict(target_spatial_head_state_dict)
                    del target_spatial_head_state_dict

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    models.pop()

        else:
            # only support finetune motion module for AnimateDiff
            def load_model_hook(models, input_dir):
                target_motion_module = MotionAdapter.from_pretrained(
                    os.path.join(input_dir, "target_motion_modules")
                )
                target_unet.load_motion_modules(target_motion_module)
                del target_motion_module

                student_motion_module = MotionAdapter.from_pretrained(
                    os.path.join(input_dir, "motion_modules")
                )
                unet_ = accelerator.unwrap_model(unet)
                unet_.load_motion_modules(student_motion_module)
                del student_motion_module

                state_dict = load_file(
                    os.path.join(
                        input_dir,
                        "discriminator",
                        "diffusion_pytorch_model.safetensors",
                    )
                )
                disc_ = accelerator.unwrap_model(discriminator)
                disc_.load_state_dict(state_dict)
                del state_dict

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # # load diffusers style into model
                    # load_model = UNet3DConditionModel.from_pretrained(
                    #     input_dir, subfolder="unet"
                    # )
                    # model.register_to_config(**load_model.config)

                    # model.load_state_dict(load_model.state_dict())
                    # del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if (
        args.enable_xformers_memory_efficient_attention
        and args.base_model_name != "animatediff"
    ):
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            if not args.use_lora:
                target_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers is not available. Make sure it is installed correctly"
            )
            # raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    if args.cd_target in ["learn", "hlearn"]:
        unet_params = list(unet.parameters()) + list(spatial_head.parameters())
    else:
        unet_params = unet.parameters()

    optimizer = optimizer_class(
        unet_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    disc_optimizer = optimizer_class(
        discriminator.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.disc_adam_beta1, args.disc_adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True
    ):
        prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train
        )
        return prompt_embeds

    WEBVID_DATA_SIZE = 2467378

    dataset = Text2VideoDataset(
        args.dataset_path,
        num_train_examples=args.max_train_samples or WEBVID_DATA_SIZE,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,
        duration=args.num_frames,
        frame_interval=args.frame_interval,
        frame_sel=args.frame_sel,
        resolution=args.resolution,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=True,
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[0.5, 0.5, 0.5],
    )
    train_dataloader = dataset.train_dataloader

    if args.disc_gt_data == "webvid":
        disc_gt_dataloader = None
    elif args.disc_gt_data in ["laion", "disney", "realisticvision", "toonyou"]:
        if args.disc_gt_data == "laion":
            from dataset.laion_dataset_wbd_modified import Text2ImageDataset, cycle

            num_image_train_examples = (
                int(args.max_train_samples * args.disc_tsn_num_frames)
                if args.max_train_samples is not None
                else int(WEBVID_DATA_SIZE * args.disc_tsn_num_frames)
            )
        else:
            from dataset.custom_dataset_wbd import Text2ImageDataset, cycle

            num_image_train_examples = min(
                (
                    int(args.max_train_samples * args.disc_tsn_num_frames)
                    if args.max_train_samples is not None
                    else int(WEBVID_DATA_SIZE * args.disc_tsn_num_frames)
                ),
                478976,
            )

        disc_gt_dataset = Text2ImageDataset(
            args.disc_gt_data_path,
            num_train_examples=num_image_train_examples,
            per_gpu_batch_size=int(args.train_batch_size * args.disc_tsn_num_frames),
            global_batch_size=int(
                args.train_batch_size
                * accelerator.num_processes
                * args.disc_tsn_num_frames
            ),
            num_workers=args.dataloader_num_workers,
            resolution=args.resolution,
            shuffle_buffer_size=1000,
            pin_memory=True,
            persistent_workers=True,
            pixel_mean=[0.5, 0.5, 0.5],
            pixel_std=[0.5, 0.5, 0.5],
        )
        disc_gt_dataloader = disc_gt_dataset.train_dataloader
        disc_gt_dataloader = cycle(disc_gt_dataloader)
    else:
        raise ValueError(
            f"Discriminator ground truth data {args.disc_gt_data} is not supported."
        )

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    disc_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=disc_optimizer,
        num_warmup_steps=args.lr_warmup_steps if args.disc_start_step == 0 else 0,
        num_training_steps=args.max_train_steps - args.disc_start_step,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    if args.cd_target in ["learn", "hlearn"]:
        (
            unet,
            spatial_head,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        ) = accelerator.prepare(
            unet,
            spatial_head,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        )
    else:
        (
            unet,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        ) = accelerator.prepare(
            unet,
            discriminator,
            optimizer,
            disc_optimizer,
            lr_scheduler,
            disc_lr_scheduler,
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        train_dataloader.num_batches / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # remove list objects to avoid bug in tensorboard
        tracker_config = {
            k: v for k, v in vars(args).items() if not isinstance(v, list)
        }
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.tracker_run_name}},
        )

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # 16. Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(
        f"  Num learnable parameters = {sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1e6} M"
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [
                d
                for d in dirs
                if (d.startswith("checkpoint") and "step" not in d and "final" not in d)
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(os.path.join(args.output_dir, path)):
                accelerator.load_state(os.path.join(args.output_dir, path))
            else:
                accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Record initial outputs
    if (not args.debug) and args.resume_from_checkpoint is None:
        log_validation(
            vae,
            deepcopy(unet),
            args,
            accelerator,
            weight_dtype,
            0,
            "online-lcm2",
            scheduler="lcm",
            num_inference_steps=2,
            add_to_trackers=True,
            use_lora=True if args.use_lora else False,
        )
        log_validation(
            vae,
            deepcopy(unet),
            args,
            accelerator,
            weight_dtype,
            0,
            "online-lcm1",
            scheduler="lcm",
            num_inference_steps=1,
            add_to_trackers=True,
            use_lora=True if args.use_lora else False,
        )
        log_validation(
            vae,
            deepcopy(teacher_unet),
            args,
            accelerator,
            weight_dtype,
            0,
            f"teacher-ddim{args.num_ddim_timesteps}",
            scheduler="ddim",
            num_inference_steps=args.num_ddim_timesteps,
            add_to_trackers=True,
            use_lora=False,
            guidance_scale=7.5,
        )

    gc.collect()
    torch.cuda.empty_cache()

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    last_update_r1_step = global_step

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(discriminator):
                # 1. Load and process the image and text conditioning
                video, text = batch["video"], batch["text"]

                video = video.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(text)

                pixel_values = video.to(dtype=weight_dtype)
                if vae.dtype != weight_dtype:
                    vae.to(dtype=weight_dtype)

                # encode pixel values with batch size of at most args.vae_encode_batch_size
                pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")
                latents = []
                for i in range(0, pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(
                            pixel_values[i : i + args.vae_encode_batch_size]
                        ).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = rearrange(
                    latents,
                    "(b t) c h w -> b c t h w",
                    b=args.train_batch_size,
                    t=args.num_frames,
                )

                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                bsz = latents.shape[0]

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = (
                    noise_scheduler.config.num_train_timesteps
                    // args.num_ddim_timesteps
                )
                index = torch.randint(
                    0, args.num_ddim_timesteps, (bsz,), device=latents.device
                ).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(
                    timesteps < 0, torch.zeros_like(timesteps), timesteps
                )

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [
                    append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]
                ]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample a random guidance scale w from U[w_min, w_max] and embed it
                # Note that for LCM-LoRA distillation it is not necessary to use a guidance scale embedding
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                if not args.use_lora:
                    w_embedding = guidance_scale_embedding(
                        w, embedding_dim=time_cond_proj_dim
                    )
                    w_embedding = w_embedding.to(
                        device=latents.device, dtype=latents.dtype
                    )
                w = w.reshape(bsz, 1, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)

                # if use predicted x_0, use the caption from the disc gt dataset
                # instead of from WebVid
                use_pred_x0 = False
                if global_step >= args.disc_start_step and not args.no_disc:
                    if args.cd_pred_x0_portion >= 0:
                        use_pred_x0 = random.random() < args.cd_pred_x0_portion

                    if args.disc_gt_data == "webvid":
                        pass
                    else:
                        gt_sample, gt_sample_caption = next(disc_gt_dataloader)
                        if use_pred_x0 and args.disc_same_caption:
                            text = gt_sample_caption
                            encoded_text = compute_embeddings_fn(text)

                # get CLIP embeddings, which is used for the adversarial loss
                with torch.no_grad():
                    clip_text_token = open_clip_tokenizer(text).to(accelerator.device)
                    clip_emb = open_clip_model.encode_text(clip_text_token)

                # 5. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")

                # 6. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                if use_pred_x0:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=weight_dtype):
                            noise = torch.randn_like(latents)
                            last_timestep = solver.ddim_timesteps[-1].unsqueeze(0)
                            last_timestep = last_timestep.repeat(bsz)
                            if args.use_lora:
                                x_0_noise_pred = unet(
                                    noise.float(),
                                    last_timestep,
                                    timestep_cond=None,
                                    encoder_hidden_states=prompt_embeds.float(),
                                ).sample
                            else:
                                x_0_noise_pred = target_unet(
                                    noise.float(),
                                    last_timestep,
                                    timestep_cond=w_embedding,
                                    encoder_hidden_states=prompt_embeds.float(),
                                ).sample
                            latents = get_predicted_original_sample(
                                x_0_noise_pred,
                                last_timestep,
                                noise,
                                noise_scheduler.config.prediction_type,
                                alpha_schedule,
                                sigma_schedule,
                            )

                noise = torch.randn_like(latents)
                noisy_model_input_list = []
                for b_idx in range(bsz):
                    if index[b_idx] != args.num_ddim_timesteps - 1:
                        noisy_model_input = noise_scheduler.add_noise(
                            latents[b_idx, None],
                            noise[b_idx, None],
                            start_timesteps[b_idx, None],
                        )
                    else:
                        # hard swap input to pure noise to ensure zero terminal SNR
                        noisy_model_input = noise[b_idx, None]
                    noisy_model_input_list.append(noisy_model_input)
                noisy_model_input = torch.cat(noisy_model_input_list, dim=0)

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=None if args.use_lora else w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    # added_cond_kwargs=encoded_text,
                ).sample

                pred_x_0_stu = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = (
                    c_skip_start * noisy_model_input + c_out_start * pred_x_0_stu
                )

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                        ).sample
                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        uncond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        ).sample
                        uncond_pred_x0 = get_predicted_original_sample(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_noise = get_predicted_noise(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                        # print(f"cond_pred_x0: {cond_pred_x0.shape}; uncond_pred_x0: {uncond_pred_x0.shape}; cond_pred_noise: {cond_pred_noise.shape}; uncond_pred_noise: {uncond_pred_noise.shape}; w: {w.shape}")
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (
                            cond_pred_noise - uncond_pred_noise
                        )
                        # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                        # augmented PF-ODE trajectory (solving backward in time)
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                # Note that we do not use a separate target network for LCM-LoRA distillation.
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        if args.use_lora:
                            target_noise_pred = unet(
                                x_prev.float(),
                                timesteps,
                                timestep_cond=None,
                                encoder_hidden_states=prompt_embeds.float(),
                            ).sample
                        else:
                            target_noise_pred = target_unet(
                                x_prev.float(),
                                timesteps,
                                timestep_cond=w_embedding,
                                encoder_hidden_states=prompt_embeds.float(),
                            ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                # 10. Calculate the CD loss and discriminator loss
                loss_dict = {}

                # 10.1. Calculate CD loss
                model_pred_cd = prepare_cd_target(model_pred, args.cd_target)
                if args.cd_target in ["learn", "hlearn"]:
                    model_pred_cd = spatial_head(model_pred_cd)
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weight_dtype):
                        target_cd = prepare_cd_target(target.float(), args.cd_target)
                        if args.cd_target in ["learn", "hlearn"]:
                            target_cd = target_spatial_head(target_cd.float())
                if args.loss_type == "l2":
                    loss_unet_cd = F.mse_loss(
                        model_pred_cd.float(), target_cd.float(), reduction="mean"
                    )
                elif args.loss_type == "huber":
                    loss_unet_cd = torch.mean(
                        torch.sqrt(
                            (model_pred_cd.float() - target_cd.float()) ** 2
                            + args.huber_c**2
                        )
                        - args.huber_c
                    )
                loss_dict["loss_unet_cd"] = loss_unet_cd
                loss_unet_total = loss_unet_cd

                # 10.2. Calculate discriminator loss
                if global_step >= args.disc_start_step and not args.no_disc:
                    model_pred_pixel = sample_and_decode(
                        model_pred,
                        vae,
                        args.num_frames,
                        args.disc_tsn_num_frames,
                        weight_dtype,
                    )

                    clip_emb = repeat(
                        clip_emb, "b n -> b t n", t=args.disc_tsn_num_frames
                    )
                    clip_emb = rearrange(clip_emb, "b t n -> (b t) n")

                    gen_dino_features, gen_sample = get_dino_features(
                        model_pred_pixel,
                        normalize_fn=normalize_fn,
                        dino_model=dino,
                        dino_hooks=args.disc_dino_hooks,
                        return_cls_token=False,
                    )
                    disc_pred_gen = discriminator(
                        gen_dino_features, clip_emb, return_key_list=["logits"]
                    )["logits"]

                    if args.disc_loss_type == "bce":
                        pos_label = torch.ones_like(disc_pred_gen)
                        loss_unet_adv = F.binary_cross_entropy_with_logits(
                            disc_pred_gen, pos_label
                        )
                    elif args.disc_loss_type == "hinge":
                        loss_unet_adv = -disc_pred_gen.mean() + 1
                    elif args.disc_loss_type == "wgan":
                        loss_unet_adv = -disc_pred_gen.mean()
                    else:
                        raise ValueError(
                            f"Discriminator loss type {args.disc_loss_type} not supported."
                        )

                    loss_dict["loss_unet_adv"] = loss_unet_adv
                    loss_unet_total = (
                        loss_unet_total + args.disc_loss_weight * loss_unet_adv
                    )

                loss_dict["loss_unet_total"] = loss_unet_total

                # 11. Backpropagate on the online student model (`unet`)
                accelerator.backward(loss_unet_total)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 12. Train the discriminator
                if global_step >= args.disc_start_step and not args.no_disc:
                    disc_optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        gen_sample = sample_and_decode(
                            model_pred.detach(),
                            vae,
                            args.num_frames,
                            args.disc_tsn_num_frames,
                            weight_dtype,
                        )

                    # get GT samples
                    if args.disc_gt_data == "webvid":
                        pixel_values = rearrange(
                            pixel_values, "(b t) c h w -> b c t h w", t=args.num_frames
                        )
                        tsn_sample_indices = tsn_sample(
                            args.num_frames, args.disc_tsn_num_frames
                        )
                        pixel_values = pixel_values[:, :, tsn_sample_indices]
                        gt_sample = rearrange(pixel_values, "b c t h w -> (b t) c h w")
                        gt_sample_clip_emb = clip_emb
                    else:
                        gt_sample = gt_sample.to(
                            accelerator.device, dtype=weight_dtype, non_blocking=True
                        )
                        with torch.no_grad():
                            gt_sample_clip_text_token = open_clip_tokenizer(
                                gt_sample_caption
                            ).to(accelerator.device)
                            gt_sample_clip_emb = open_clip_model.encode_text(
                                gt_sample_clip_text_token
                            )

                    # get discriminator predictions on generated sampels
                    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
                        gen_dino_features, gen_sample = get_dino_features(
                            gen_sample,
                            normalize_fn=normalize_fn,
                            dino_model=dino,
                            dino_hooks=args.disc_dino_hooks,
                            return_cls_token=False,
                        )
                    disc_pred_gen = discriminator(
                        gen_dino_features, clip_emb, return_key_list=["logits"]
                    )["logits"]

                    # get discriminator predictions on GT samples
                    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
                        gt_dino_features, processed_gt_sample = get_dino_features(
                            gt_sample,
                            normalize_fn=normalize_fn,
                            dino_model=dino,
                            dino_hooks=args.disc_dino_hooks,
                            return_cls_token=False,
                        )
                    disc_pred_gt = discriminator(
                        gt_dino_features, gt_sample_clip_emb, return_key_list=["logits"]
                    )["logits"]

                    if args.disc_loss_type == "bce":
                        pos_label = torch.ones_like(disc_pred_gen)
                        neg_label = torch.zeros_like(disc_pred_gen)
                        loss_disc_gt = F.binary_cross_entropy_with_logits(
                            disc_pred_gt, pos_label
                        )
                        loss_disc_gen = F.binary_cross_entropy_with_logits(
                            disc_pred_gen, neg_label
                        )
                    elif args.disc_loss_type == "hinge":
                        loss_disc_gt = (
                            torch.max(torch.zeros_like(disc_pred_gt), 1 - disc_pred_gt)
                        ).mean()
                        loss_disc_gen = (
                            torch.max(torch.zeros_like(disc_pred_gt), 1 + disc_pred_gen)
                        ).mean()
                    elif args.disc_loss_type == "wgan":
                        loss_disc_gt = (
                            torch.max(-torch.ones_like(disc_pred_gt), -disc_pred_gt)
                        ).mean()
                        loss_disc_gen = (
                            torch.max(-torch.ones_like(disc_pred_gt), disc_pred_gen)
                        ).mean()
                    else:
                        raise ValueError(
                            f"Discriminator loss type {args.disc_loss_type} not supported."
                        )

                    loss_disc_total = loss_disc_gt + loss_disc_gen
                    loss_dict["loss_disc_gt"] = loss_disc_gt
                    loss_dict["loss_disc_gen"] = loss_disc_gen

                    if args.disc_lambda_r1 > 0:
                        # not sure if this is the correct way to calculate the gradient penalty
                        with torch.autocast("cuda", dtype=weight_dtype):
                            alpha = torch.rand(
                                int(bsz * args.disc_tsn_num_frames),
                                1,
                                1,
                                1,
                                device=gt_sample.device,
                            )
                            interpolations = (
                                alpha * processed_gt_sample + (1 - alpha) * gen_sample
                            )
                            interpolation_features, interpolations = get_dino_features(
                                interpolations,
                                normalize_fn=normalize_fn,
                                dino_model=dino,
                                dino_hooks=args.disc_dino_hooks,
                                return_cls_token=False,
                                preprocess_sample=False,
                            )
                            for feat_idx in range(len(interpolation_features)):
                                interpolation_features[feat_idx].requires_grad = True

                            disc_interpolation_logit_list = discriminator(
                                interpolation_features,
                                clip_emb,
                                return_key_list=["logit_list"],
                            )["logit_list"]

                            gradients = torch.autograd.grad(
                                outputs=[
                                    item.sum() for item in disc_interpolation_logit_list
                                ],
                                inputs=interpolation_features,
                                create_graph=True,
                            )
                            gradients = torch.cat(gradients, dim=1)
                            gradients = gradients.reshape(gradients.size(0), -1)
                            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        # adjust disc_lambda_r1
                        if global_step - last_update_r1_step > 500:
                            if grad_penalty >= 100:
                                args.disc_lambda_r1 = args.disc_lambda_r1 * 5.0
                                last_update_r1_step = global_step
                                logger.warning(
                                    f"Graident penalty too high, increasing disc_lambda_r1 to {args.disc_lambda_r1}"
                                )

                        loss_dict["loss_disc_r1"] = grad_penalty
                        loss_disc_total = (
                            loss_disc_total + args.disc_lambda_r1 * grad_penalty
                        )

                    loss_dict["loss_disc_total"] = loss_disc_total

                    accelerator.backward(loss_disc_total)
                    disc_optimizer.step()
                    disc_lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 13. Make EMA update to target student model parameters (`target_unet`)
                if not args.use_lora:
                    update_ema(
                        target_unet.parameters(), unet.parameters(), args.ema_decay
                    )
                if args.cd_target in ["learn", "hlearn"]:
                    update_ema(
                        target_spatial_head.parameters(),
                        spatial_head.parameters(),
                        args.ema_decay,
                    )
                progress_bar.update(1)
                global_step += 1

                # according to https://github.com/huggingface/diffusers/issues/2606
                # DeepSpeed need to run save for all processes
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d
                                for d in checkpoints
                                if (
                                    d.startswith("checkpoint")
                                    and "step" not in d
                                    and "final" not in d
                                )
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    try:
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    except Exception as e:
                        logger.info(f"Failed to save state: {e}")

                if global_step % args.validation_steps == 0:
                    if global_step >= args.disc_start_step and not args.no_disc:
                        try:
                            # gt sample: in [-1, 1], shape: b, c, h, w
                            gt_sample = gt_sample.detach()
                            gt_sample = gt_sample.add(1).div(2)
                            gt_sample = rearrange(gt_sample, "b c h w -> b h w c")
                            gt_sample_list = []
                            for i in range(gt_sample.shape[0]):
                                gt_sample_list.append(gt_sample[i].cpu().numpy())
                        except:
                            gt_sample_list = None
                    else:
                        gt_sample_list = None

                    log_validation(
                        vae,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        "online",
                        use_lora=args.use_lora,
                        num_inference_steps=1,
                        disc_gt_images=gt_sample_list,
                    )
                    log_validation(
                        vae,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                        "online",
                        use_lora=args.use_lora,
                        num_inference_steps=2,
                    )
                    if args.cd_target in ["learn", "hlearn"]:
                        log_validation(
                            vae,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "online",
                            use_lora=args.use_lora,
                            num_inference_steps=1,
                            spatial_head=target_spatial_head,
                            logger_prefix="(head) ",
                        )
                        log_validation(
                            vae,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "online",
                            use_lora=args.use_lora,
                            num_inference_steps=2,
                            spatial_head=target_spatial_head,
                            logger_prefix="(head) ",
                        )
                    if not args.use_lora:
                        log_validation(
                            vae,
                            target_unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "target",  # will not be used if args.use_lora
                            use_lora=args.use_lora,
                            num_inference_steps=1,
                        )
                        log_validation(
                            vae,
                            target_unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "target",  # will not be used if args.use_lora
                            use_lora=args.use_lora,
                            num_inference_steps=2,
                        )
                    accelerator.wait_for_everyone()

            logs = {
                "unet_lr": lr_scheduler.get_last_lr()[0],
                "disc_lr": disc_lr_scheduler.get_last_lr()[0],
                "disc_r1_weight": args.disc_lambda_r1,
            }
            for loss_name, loss_value in loss_dict.items():
                if type(loss_value) == torch.Tensor:
                    logs[loss_name] = loss_value.item()
                else:
                    logs[loss_name] = loss_value

            current_time = datetime.now().strftime("%m-%d-%H:%M")
            progress_bar.set_postfix(
                **logs,
                **{"cur time": current_time},
                **{"video_name": batch["__key__"]},
            )
            try:
                accelerator.log(logs, step=global_step)
            except Exception as e:
                logger.info(f"Failed to log metrics at step {global_step}: {e}")

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_lora:
            unet.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))
            lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
            StableDiffusionPipeline.save_lora_weights(
                os.path.join(args.output_dir, "checkpoint-final", "unet_lora"),
                lora_state_dict,
            )
            if args.cd_target in ["learn", "hlearn"]:
                spatial_head_ = accelerator.unwrap_model(spatial_head)
                spatial_head_.save_pretrained(
                    os.path.join(args.output_dir, "spatial_head")
                )
                target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
                target_spatial_head_.save_pretrained(
                    os.path.join(args.output_dir, "target_spatial_head")
                )
        else:
            # save motion module
            unet_ = accelerator.unwrap_model(unet)
            unet_.save_motion_modules(os.path.join(args.output_dir, "motion_modules"))
            target_unet = accelerator.unwrap_model(target_unet)
            target_unet.save_motion_modules(
                os.path.join(args.output_dir, "target_motion_modules")
            )
            if args.cd_target in ["learn", "hlearn"]:
                spatial_head_ = accelerator.unwrap_model(spatial_head)
                spatial_head_.save_pretrained(
                    os.path.join(args.output_dir, "spatial_head")
                )
                target_spatial_head_ = accelerator.unwrap_model(target_spatial_head)
                target_spatial_head_.save_pretrained(
                    os.path.join(args.output_dir, "target_spatial_head")
                )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
