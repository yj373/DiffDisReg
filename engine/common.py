import torch
import os
import math
import copy
import argparse
import sys
sys.path.append("./")
import numpy as np
from accelerate.logging import get_logger
from diffusers import (
    DPMSolverMultistepScheduler, 
    DiffusionPipeline, 
    StableDiffusionPipeline, 
    DDPMScheduler, 
    AutoencoderKL, 
    UNet2DConditionModel,
)
from diffusers.utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.optimization import get_scheduler
from transformers import PretrainedConfig
from transformers import AutoTokenizer
from transformers import CLIPTextModel
from huggingface_hub.utils import insecure_hashlib
from huggingface_hub import create_repo
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from data.DreamBooth import PromptDataset, DreamBoothDataset, collate_fn

BASE_MODEL_MAP = {
    "SD1_4": "CompVis/stable-diffusion-v1-4",
    "SD1_5": "stable-diffusion-v1-5/stable-diffusion-v1-5"
}

def save_model_card(
    repo_id: str,
    images=None,
    base_model_key=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
    lora=False
):
    img_str = ""
    base_model = BASE_MODEL_MAP[base_model_key]
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    if lora:
        model_description = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
        tags = ["text-to-image", "diffusers", "lora", "diffusers-training"]    
    else:
        model_description = f"""
# DreamBooth - {repo_id}

This is a dreambooth model derived from {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/).
You can find some example images in the following. \n
{img_str}

DreamBooth for the text encoder was enabled: {train_text_encoder}.
""" 
        tags = ["text-to-image", "dreambooth", "diffusers-training"]

    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(["stable-diffusion", "stable-diffusion-diffusers"])
    else:
        tags.extend(["if", "if-diffusers"])

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


if is_wandb_available():
    import wandb

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger = get_logger(__name__)
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    if args.validation_images is None:
        images = []
        for _ in range(args.num_validation_images):
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, generator=generator).images[0]
                images.append(image)
    else:
        images = []
        for image in args.validation_images:
            image = Image.open(image)
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def generate_class_images(args, accelerator, logger):
    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < args.num_class_images:
        # print(cur_class_images, args.num_class_images, class_images_dir)
        torch_dtype = torch.bfloat16 if accelerator.device.type == "cuda" else torch.float32
        if args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)

        for example in tqdm(
            sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
        ):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                # print(os.path.abspath(image_filename))
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def model_prepare(args, accelerator):
    # Handle the repository creation
    repo_id = None
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )

    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    return tokenizer, noise_scheduler, text_encoder_cls, text_encoder, vae, unet, repo_id


def training_prepare(
    args,
    accelerator, 
    tokenizer, 
    pre_computed_encoder_hidden_states, 
    pre_computed_class_prompt_encoder_hidden_states,
    unet,
    text_encoder,
    optimizer,
):
    single_class_token = True
    if args.with_prior_preservation and args.class_data_dir == 'auto' and args.num_reg_concepts > 0:
        single_class_token = False
    if args.with_prior_preservation and len(args.class_data_dir.split('/')[-1].split('_')) > 1:
        single_class_token = False
    print(f">>> single_class_token: {single_class_token}")
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_seg_root=args.instance_seg_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_seg_root=args.class_seg_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
        single_class_token=single_class_token,
        # augmentation= DreamBoothAugment(op_num=1, p=args.augment_p) if args.augmentation else None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth-lora", config=tracker_config)

    return train_dataset, train_dataloader, unet, text_encoder, optimizer, train_dataloader, lr_scheduler, num_update_steps_per_epoch
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Flag to evaluate with LoRA.",
    )
    parser.add_argument(
        "--finetune_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model to finetune.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step number of the model to evaluate.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the model to evaluate.",
    )
    parser.add_argument(
        "--prompt1",
        type=str,
        required=True,
        help="First prompt to evaluate.",
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        required=True,
        help="Second prompt to evaluate.",
    )
    args = parser.parse_args()
    if args.lora:
        # print('>>> loading pipeline')
        pipeline = DiffusionPipeline.from_pretrained(
            args.finetune_model, dtype=torch.float32, safety_checker=None,
        ).to("cuda:0")
        # print('>>> loading lora weight')
        if args.step > 0 :
            pipeline.load_lora_weights(args.root, subfolder="checkpoint-{}".format(args.step), weight_name="pytorch_lora_weights.safetensors")
        else:
            pipeline.load_lora_weights(args.root, weight_name="pytorch_lora_weights.safetensors")
        pipeline.set_progress_bar_config(disable=True)
        # print('>>>')
    else:
        if args.step > 0 :
            unet = UNet2DConditionModel.from_pretrained(args.root, subfolder="checkpoint-{}/unet".format(args.step))
            text_encoder = CLIPTextModel.from_pretrained(args.root, subfolder="text_encoder")
            pipeline = DiffusionPipeline.from_pretrained(
                args.finetune_model, unet=unet, text_encoder=text_encoder, dtype=torch.float32, safety_checker=None
            ).to("cuda:0")
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.finetune_model, dtype=torch.float32, safety_checker=None
            ).to("cuda:0")
        pipeline.set_progress_bar_config(disable=True)

    tokenizer = AutoTokenizer.from_pretrained(
            args.finetune_model,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
    )
    text_inputs_1 = tokenize_prompt(tokenizer, args.prompt1, tokenizer_max_length=None)
    prompt_embeds_1 = encode_prompt(
                    pipeline.text_encoder,
                    text_inputs_1.input_ids,
                    text_inputs_1.attention_mask,
                    text_encoder_use_attention_mask=False,
    ).flatten()
    text_inputs_2 = tokenize_prompt(tokenizer, args.prompt2, tokenizer_max_length=None)
    prompt_embeds_2 = encode_prompt(
                    pipeline.text_encoder,
                    text_inputs_2.input_ids,
                    text_inputs_2.attention_mask,
                    text_encoder_use_attention_mask=False,
    ).flatten()
    print("Prompt 1: ", prompt_embeds_1, prompt_embeds_1.shape)
    print("Prompt 2: ", prompt_embeds_2, prompt_embeds_2.shape)
    sim = torch.nn.functional.cosine_similarity(prompt_embeds_1, prompt_embeds_2, dim=0)
    print("Cosine similarity: ", sim, sim.shape)
    