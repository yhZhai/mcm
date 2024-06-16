import argparse
import logging
import os

logger = logging.getLogger(__name__)


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="animatediff",
        choices=["animatediff", "modelscope"],
        help="The name of the base model to use.",
    )
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        nargs="+",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--from_pretrained_unet",
        type=str,
        default=None,
        help="Only load the parameters from a pretrained UNet.",
    )
    parser.add_argument(
        "--from_pretrained_disc",
        type=str,
        default=None,
        help="Only load the parameters from a pretrained discriminator.",
    )
    # ----Image Processing----
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help=("The number of frames for the snippet."),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="bilinear",
        help=(
            "The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`,"
            " `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        default=[],
        help=("The dataset root path for webvid."),
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="",
        help=("The caption tsv file path."),
    )
    parser.add_argument(
        "--frame_sel",
        type=str,
        default="random",
        choices=["random", "first"],
        help="The frame selection method.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="The frame interval for frame selection.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use. Assume a batch size of 128.",
    )
    parser.add_argument(
        "--disc_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use. Assume a batch size of 128.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--disc_adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--disc_adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # parser.add_argument(
    #     "--max_skip_steps",
    #     type=int,
    #     default=1,
    #     help="The maximum number of steps to skip for ODE solvers.",
    # )
    parser.add_argument(
        "--zero_snr",
        action="store_true",
        help=("Whether to rescale betas to enable zero terminal SNR."),
    )
    parser.add_argument(
        "--beta_schedule",
        default="scaled_linear",
        type=str,
        help="The schedule to use for the beta values.",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=32,
        required=False,
        help=(
            "The batch size used when encoding (and decoding) images to latents (and vice versa) using the VAE."
            " Encoding or decoding the whole batch at once may run into OOM issues."
        ),
    )
    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help=(
            "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        ),
    )
    parser.add_argument(
        "--cd_target",
        type=str,
        default="raw",
        choices=[
            "raw",
            "diff",
            "freql",
            "freqh",
            "learn",
            "hlearn",
            "lcor",
            "gcor",
            "sgcor",
            "sgcord",
        ],
        help=(
            "The loss target for consistency distillation."
            " raw: use the raw latent;"
            " diff: use latent difference;"
            " freql: use latent low-frequency component;"
            " freqh: use latent high-frequency component;"
            " learn: use light-weight learnable spatial head;"
            " hlearn: use heavy-weight learnable spatial head;"
            " lcor: use latent local correlation;"
            " gcor: use latent global correlation;"
            " sgcor: use latent scaled global correlation;"
        ),
    )
    parser.add_argument(
        "--spatial_cd_weight",
        type=float,
        default=0.0,
        help="The weight for the spatial consistency distillation.",
    )
    parser.add_argument(
        "--cd_pred_x0_portion",
        type=float,
        default=0.0,
        help="The portion to use predicted x0 latent for the consistency distillation.",
    )
    # ----LoRA----
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether or not to use LoRA for the latent consistency distillation.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help=(
            "The value of the LoRA alpha parameter, which controls the scaling factor in front of the LoRA weight"
            " update delta_W. No scaling will be performed if this value is equal to `lora_rank`."
        ),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "A comma-separated string of target module keys to add LoRA to. If not set, a default list of modules will"
            " be used. By default, LoRA will be applied to all conv and linear layers."
        ),
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="video lcm",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default="experiment",
        help="The `run_name` argument passed to Accelerator.init_trackers., used specifically for wandb",
    )
    # ----------Discriminator Arguments----------
    parser.add_argument(
        "--no_disc",
        action="store_true",
        default=False,
        help="Do not add the adversarial loss.",
    )
    parser.add_argument(
        "--disc_loss_type",
        default="hinge",
        type=str,
        choices=["bce", "hinge", "wgan"],
        help="Loss type for adv. loss.",
    )
    parser.add_argument(
        "--disc_loss_weight", default=1.0, type=float, help="Loss weight for adv. loss."
    )
    parser.add_argument(
        "--disc_tsn_num_frames",
        default=2,
        type=int,
        help="Number of sampling frames for adv. loss.",
    )
    parser.add_argument(
        "--disc_lambda_r1", default=0, type=float, help="R1 regularization weight."
    )
    parser.add_argument(
        "--disc_dino_hooks",
        type=int,
        default=[2, 5, 8, 11],
        nargs="+",
        help="DINO hooks.",
    )
    parser.add_argument(
        "--disc_start_step",
        type=int,
        default=0,
        help="The start step to add the discriminator.",
    )
    parser.add_argument(
        "--disc_gt_data",
        type=str,
        default="webvid",
        choices=["webvid", "laion", "disney", "realisticvision", "toonyou"],
        help="The ground truth data for discriminator.",
    )
    parser.add_argument(
        "--disc_gt_data_path",
        type=str,
        default="",
        help="The ground truth data path for discriminator.",
    )
    parser.add_argument(
        "--disc_same_caption",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "If True, use the same caption for the discriminator and the generator. "
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    assert not (
        args.base_model_name is None and args.pretrained_teacher_model is None
    ), "You must specify either `--base_model_name` or `--pretrained_teacher_model`."

    if args.base_model_name == "animatediff":
        args.pretrained_teacher_model = (
            "yuanhao_project/diffusers/stable-diffusion-v1-5"
        )
        args.motion_adapter_path = (
            "yuanhao_project/diffusers/animatediff-motion-adapter-v1-5-2"
        )
        args.online_pretrained_teacher_model = "runwayml/stable-diffusion-v1-5"
        args.online_motion_adapter_path = "guoyww/animatediff-motion-adapter-v1-5-2"
        logging.info(
            "Using the `animatediff` base model. The `--pretrained_teacher_model` will be set to"
            " `runwayml/stable-diffusion-v1-5`, and the `--motion_adapter_path` will be set to"
            " `guoyww/animatediff-motion-adapter-v1-5-2`."
        )
        # raise NotImplementedError("check save model and load model hook and xformers")
    elif args.base_model_name == "modelscope":
        args.pretrained_teacher_model = (
            "yuanhao_project/diffusers/text-to-video-ms-1.7b"
        )
        args.motion_adapter_path = ""
        args.online_pretrained_teacher_model = "ali-vilab/text-to-video-ms-1.7b"
        args.online_motion_adapter_path = ""
        logging.info(
            "Using the `modelscope` base model. The `--pretrained_teacher_model` will be set to"
            " `ali-vilab/text-to-video-ms-1.7b`."
        )
    else:
        raise ValueError(f"Invalid `--base_model_name` value: {args.base_model_name}")

    if args.disc_same_caption:
        args.disc_tsn_num_frames = 1

    return args
