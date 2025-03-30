import torch
import argparse
import warnings
import os
# import sys
# sys.path.append("../")
# sys.path.append("./")
import numpy as np

from engine.LoRA import lora_train
from engine.Scratch import scratch_train
from engine.evaluation import fid_evaluation, clip_i_evaluation, clip_t_evaluation, dino_evaluation
from .evaluate import generate_images_for_evaluation

from ray import train

from transformers import AutoTokenizer
from transformers import CLIPTextModel
from transformers import BertModel, BertTokenizer

CLASS_MAP = {
    "backpack": False,
    "teapot": False,
    "vase": False,
    "wolf plushie": False,
    "bear plushie": False,
    "robot toy": False,
    "shiny sneaker": False,
    "clock": False,
    "can": False,
    "candle": False,
    "berry bowl": False,
    "dog": True,
    "cat": True,
}

LIVE_OBJS = [
    'dog',
    'monkey',
    'rabbit',
    'bird',
    'cat',
    'person',
    'butterfly',
    'chicken',
    'elephant',
    'horse',
    'bear',
    'cow',
    'tree',
    'frog',
    'goat',
    'mouse',
    'mushroom',
    'pig',
    'fox',
    'snake',
    'deer',
]

OBJS = [
    'vase', 
    'teapot', 
    'backpack', 
    'bowl', 
    'car', 
    'bike', 
    'basketball', 
    'hat', 
    'house', 
    'boat', 
    'book',
    'headphone',
    'jacket',
    'lamp',
    'plushie',
    'ring',
    'sneaker',
    'sofa',
    'table',
    'toy',
    'bottle',
]


def get_args(input_args=None):
    
    parser = argparse.ArgumentParser(description='DiffDisReg')
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_seg_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the segmentation masks of training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="",
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_seg_dir",
        type=str,
        default="",
        required=False,
        help="A folder containing the training data of class images (segmentation masks).",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--skip_save_text_encoder", action="store_true", required=False, help="Set to not save text encoder"
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight", 
        type=float, 
        default=1.0, 
        help="The weight of prior preservation loss."
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
        help="Flag to train with LoRA.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
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
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Whether or not to disable progress bar",
    )

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=8, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if use_lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )
    parser.add_argument(
        "--lora_text_encoder_r",
        type=int,
        default=8,
        help="Lora rank for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_alpha",
        type=int,
        default=32,
        help="Lora alpha for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_dropout",
        type=float,
        default=0.0,
        help="Lora dropout for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora and `train_text_encoder` are True",
    )

    ###
    parser.add_argument(
        "--mse_weight", 
        type=float, 
        default=1.0, 
        help="The weight of MSE loss."
    )
    parser.add_argument(
        "--with_mse_snr",
        default=False,
        action="store_true",
        help="Flag to add snr to mse loss.",
    )
    parser.add_argument(
        "--mse_snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--with_attention_reg_snr",
        default=False,
        action="store_true",
        help="Flag to add snr to attention regularization.",
    )
    parser.add_argument(
        "--reg_snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--reg_snr_mode",
        type=int,
        default=0,
        help="snr weight mode: 0 -- min-gamma, 1 -- max-gamma, others -- constant"
    )
    parser.add_argument(
        "--with_attention_reg_sigmoid",
        default=False,
        action="store_true",
        help="Flag to add sigmoid weight to attention regularization.",
    )
    parser.add_argument(
        "--reg_sigmoid_delta",
        type=float,
        default=200.0,
        help="Attention regularization sigmoid delta"
    )
    parser.add_argument(
        "--reg_sigmoid_kappa",
        type=float,
        default=10.0,
        help="Attention regularization sigmoid kappa"
    )
    parser.add_argument(
        "--with_attention_reg",
        default=False,
        action="store_true",
        help="Flag to add attention regularization.",
    )
    parser.add_argument(
        "--attention_loss_weight", 
        type=float, 
        default=1.0, 
        help="The weight of ODISE regularization loss."
    )
    parser.add_argument(
        "--attention_reg_token_idx1", 
        type=int, 
        default=1, 
        help="Index of the first refgularized token in the prompt."
    )
    parser.add_argument(
        "--attention_reg_token_idx2", 
        type=int, 
        default=-1, 
        help="Index of the second regularized token in the prompt."
    )
    # always the key token
    # parser.add_argument(
    #     "--attention_reg_prior_token_idx", 
    #     type=int, 
    #     default=2, 
    #     help="Parameter for the mask construction."
    # )
    parser.add_argument("--weight_8", type=float, default=0.3, help="Weight of 8x8 attention map for the mask construction")
    parser.add_argument("--weight_16", type=float, default=0.5, help="Weight of 8x8 attention map for the mask construction")
    parser.add_argument("--weight_32", type=float, default=0.1, help="Weight of 8x8 attention map for the mask construction")
    parser.add_argument("--weight_64", type=float, default=0.1, help="Weight of 8x8 attention map for the mask construction")
    parser.add_argument(
        "--self_attn",
        default=False,
        action="store_true",
        help="Flag to store self attention maps.",
    )
    parser.add_argument("--alpha", type=float, default=8.0, help="Parameter for the mask construction")
    parser.add_argument("--beta", type=float, default=0.4, help="Parameter for the mask construction")
    parser.add_argument(
        "--disable_input_reg",
        default=False,
        action="store_true",
        help="Flag to disable input BCE loss.",
    )
    ###

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args


def clean_class_name(instance_class_name):
    i = 0
    while i < len(instance_class_name) and not instance_class_name[i].isdigit():
        i += 1
    return instance_class_name[:i]


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True


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
 

def compute_semantic_sim(tokenizer, text_encoder, concept1, concept2, mode=0, live_subject=False):
    if mode == 0:
        print(">>> CLIP")
        text_inputs_1 = tokenize_prompt(tokenizer, concept1, tokenizer_max_length=None)
        prompt_embeds_1 = encode_prompt(
            text_encoder,
            text_inputs_1.input_ids,
            text_inputs_1.attention_mask,
            text_encoder_use_attention_mask=False,
        ).flatten()
        text_inputs_2 = tokenize_prompt(tokenizer, concept2, tokenizer_max_length=None)
        prompt_embeds_2 = encode_prompt(
                    text_encoder,
                    text_inputs_2.input_ids,
                    text_inputs_2.attention_mask,
                    text_encoder_use_attention_mask=False,
        ).flatten()
        with torch.no_grad():
            sim = torch.nn.functional.cosine_similarity(prompt_embeds_1, prompt_embeds_2, dim=0)
    elif mode == 1:
        print(">>> BERT")
        model_name = '/dev/shm/alexJiang/source/BERT'
        bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)

        tokens_1 = bert_tokenizer.tokenize(concept1)
        token_id_1 = bert_tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_1 = torch.tensor([token_id_1])

        tokens_2 = bert_tokenizer.tokenize(concept2)
        token_id_2 = bert_tokenizer.convert_tokens_to_ids(tokens_2)
        input_ids_2 = torch.tensor([token_id_2])
        with torch.no_grad():
            outputs_1 = bert_model(input_ids_1)
            # print(outputs_1.last_hidden_state.shape)
            prompt_embeds_1 = outputs_1.last_hidden_state.mean(dim=1).flatten()
            outputs_2 = bert_model(input_ids_2)
            # print(outputs_2.last_hidden_state.shape)
            prompt_embeds_2 = outputs_2.last_hidden_state.mean(dim=1).flatten()
            sim = torch.nn.functional.cosine_similarity(prompt_embeds_1, prompt_embeds_2, dim=0)
    else:
        print(">>> BERT sentence")
        model_name = '/dev/shm/alexJiang/source/BERT'
        bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertModel.from_pretrained(model_name)
        if live_subject:
            ## live subject prompt
            prompt_list_1 = [
                'a {} in the jungle'.format(concept1),
                'a {} in the snow'.format(concept1),
                'a {} on the beach'.format(concept1),
                'a {} on a cobblestone street'.format(concept1),
                'a {} on top of pink fabric'.format(concept1),
                'a {} on top of a wooden floor'.format(concept1),
                'a {} with a city in the background'.format(concept1),
                'a {} with a mountain in the background'.format(concept1),
                'a {} with a blue house in the background'.format(concept1),
                'a {} on top of a purple rug in a forest'.format(concept1),
                'a {} wearing a red hat'.format(concept1),
                'a {} wearing a santa hat'.format(concept1),
                'a {} wearing a rainbow scarf'.format(concept1),
                'a {} wearing a black top hat and a monocle'.format(concept1),
                'a {} in a chef outfit'.format(concept1),
                'a {} in a firefighter outfit'.format(concept1),
                'a {} in a police outfit'.format(concept1),
                'a {} wearing pink glasses'.format(concept1),
                'a {} wearing a yellow shirt'.format(concept1),
                'a {} in a purple wizard outfit'.format(concept1),
                'a red {}'.format(concept1),
                'a purple {}'.format(concept1),
                'a shiny {}'.format(concept1),
                'a wet {}'.format(concept1),
                'a cube shaped {}'.format(concept1)
            ]
            prompt_list_2 = [
                'a {} in the jungle'.format(concept2),
                'a {} in the snow'.format(concept2),
                'a {} on the beach'.format(concept2),
                'a {} on a cobblestone street'.format(concept2),
                'a {} on top of pink fabric'.format(concept2),
                'a {} on top of a wooden floor'.format(concept2),
                'a {} with a city in the background'.format(concept2),
                'a {} with a mountain in the background'.format(concept2),
                'a {} with a blue house in the background'.format(concept2),
                'a {} on top of a purple rug in a forest'.format(concept2),
                'a {} wearing a red hat'.format(concept2),
                'a {} wearing a santa hat'.format(concept2),
                'a {} wearing a rainbow scarf'.format(concept2),
                'a {} wearing a black top hat and a monocle'.format(concept2),
                'a {} in a chef outfit'.format(concept2),
                'a {} in a firefighter outfit'.format(concept2),
                'a {} in a police outfit'.format(concept2),
                'a {} wearing pink glasses'.format(concept2),
                'a {} wearing a yellow shirt'.format(concept2),
                'a {} in a purple wizard outfit'.format(concept2),
                'a red {}'.format(concept2),
                'a purple {}'.format(concept2),
                'a shiny {}'.format(concept2),
                'a wet {}'.format(concept2),
                'a cube shaped {}'.format(concept2)
            ]
        else:
            ## object prompt
            prompt_list_1 = [
                'a {} in the jungle'.format(concept1),
                'a {} in the snow'.format(concept1),
                'a {} on the beach'.format(concept1),
                'a {} on a cobblestone street'.format(concept1),
                'a {} on top of pink fabric'.format(concept1),
                'a {} on top of a wooden floor'.format(concept1),
                'a {} with a city in the background'.format(concept1),
                'a {} with a mountain in the background'.format(concept1),
                'a {} with a blue house in the background'.format(concept1),
                'a {} on top of a purple rug in a forest'.format(concept1),
                'a {} with a wheat field in the background'.format(concept1),
                'a {} with a tree and autumn leaves in the background'.format(concept1),
                'a {} with the Eiffel Tower in the background'.format(concept1),
                'a {} floating on top of water'.format(concept1),
                'a {} floating in an ocean of milk'.format(concept1),
                'a {} on top of green grass with sunflowers around it'.format(concept1),
                'a {} on top of a mirror'.format(concept1),
                'a {} on top of the sidewalk in a crowded street'.format(concept1),
                'a {} on top of a dirt road'.format(concept1),
                'a {} on top of a white rug'.format(concept1),
                'a red {}'.format(concept1),
                'a purple {}'.format(concept1),
                'a shiny {}'.format(concept1),
                'a wet {}'.format(concept1),
                'a cube shaped {}'.format(concept1)
            ]
            prompt_list_2 = [
                'a {} in the jungle'.format(concept2),
                'a {} in the snow'.format(concept2),
                'a {} on the beach'.format(concept2),
                'a {} on a cobblestone street'.format(concept2),
                'a {} on top of pink fabric'.format(concept2),
                'a {} on top of a wooden floor'.format(concept2),
                'a {} with a city in the background'.format(concept2),
                'a {} with a mountain in the background'.format(concept2),
                'a {} with a blue house in the background'.format(concept2),
                'a {} on top of a purple rug in a forest'.format(concept2),
                'a {} with a wheat field in the background'.format(concept2),
                'a {} with a tree and autumn leaves in the background'.format(concept2),
                'a {} with the Eiffel Tower in the background'.format(concept2),
                'a {} floating on top of water'.format(concept2),
                'a {} floating in an ocean of milk'.format(concept2),
                'a {} on top of green grass with sunflowers around it'.format(concept2),
                'a {} on top of a mirror'.format(concept2),
                'a {} on top of the sidewalk in a crowded street'.format(concept2),
                'a {} on top of a dirt road'.format(concept2),
                'a {} on top of a white rug'.format(concept2),
                'a red {}'.format(concept2),
                'a purple {}'.format(concept2),
                'a shiny {}'.format(concept2),
                'a wet {}'.format(concept2),
                'a cube shaped {}'.format(concept2)
            ]
        sim_list = []
        for i, (p1, p2) in enumerate(zip(prompt_list_1, prompt_list_2)):
            tokens_1 = bert_tokenizer.tokenize(p1)
            token_id_1 = bert_tokenizer.convert_tokens_to_ids(tokens_1)
            input_ids_1 = torch.tensor([token_id_1])

            tokens_2 = bert_tokenizer.tokenize(p2)
            token_id_2 = bert_tokenizer.convert_tokens_to_ids(tokens_2)
            input_ids_2 = torch.tensor([token_id_2])
            with torch.no_grad():
                outputs_1 = bert_model(input_ids_1)
                word_embeddings_1 = outputs_1.last_hidden_state

                outputs_2 = bert_model(input_ids_2)
                word_embeddings_2 = outputs_2.last_hidden_state
                if i < 20:
                    emb_1 = word_embeddings_1.squeeze()[1, :]
                    emb_2 = word_embeddings_2.squeeze()[1, :]
                else:
                    emb_1 = word_embeddings_1.squeeze()[2, :]
                    emb_2 = word_embeddings_2.squeeze()[2, :]
                sim_list.append(torch.nn.functional.cosine_similarity(emb_1, emb_2, dim=0))
        sim = np.array(sim_list).mean()
    print(f"Cosine similarity ({concept1}, {concept2}): ", sim)
    return sim.item()


def select_reg_semantics(pretrained_model_name_or_path, instance_class_name, select=3, verbose=False, mode=0, order=-1):
    # assert order in [1, -1], f"invalid order (-1, 1): {order}"
    class_name = clean_class_name(instance_class_name)
    orig_class_name = class_name.replace('_', ' ')
    tokens = class_name.split("_")
    if len(tokens) > 1:
        class_name = tokens[-1]
        print(f">>> simplified token for selection: {class_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    live_subject=CLASS_MAP[orig_class_name]
    print(f"{orig_class_name} is a live subject: {live_subject}")
    concepts = OBJS
    if live_subject:
        concepts = LIVE_OBJS
    sims = []
    concepts_ = []
    if mode == -1:
        # default setting: live --mode 2, --order 1; reg --mode 0 --order -1
        mode = 2 if live_subject else 0
        order = 1 if live_subject else -1
    print(f"mode: {mode}, order: {order}")
    for c in concepts:
        if c == class_name:
            continue
        sims.append(compute_semantic_sim(tokenizer, text_encoder, class_name, c, mode=mode, live_subject=live_subject))
        concepts_.append(c)
    if order < 0:
        # big sim to small
        indexes = sorted(range(len(sims)), key=lambda i: -sims[i])
    else:
        indexes = sorted(range(len(sims)), key=lambda i: sims[i])
    if verbose:
        print(">>>> sorted")
        for idx in indexes:
            print(f"({class_name}, {concepts_[idx]}): {sims[idx]}")
    selected = []
    for i in indexes[:select]:
        selected.append(concepts_[i])
    
    return selected


def training_function(config, args=None):
    output_dir = args.output_dir
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    instance_data_dir = os.path.join("Dataset/DreamBooth/", args.instance_class_name)
    instance_seg_dir = os.path.join("Dataset/DreamBooth/", args.instance_class_name + "_seg")
    class_name = clean_class_name(args.instance_class_name)
    if len(args.class_data_dir) == 0 or (not os.path.isdir(args.class_data_dir)):
        class_data_dir = os.path.join("output/class_data", class_name)
        class_seg_dir = os.path.join("output/class_data_seg", class_name)
    else:
        class_data_dir = args.class_data_dir
        class_seg_dir = class_data_dir.replace("class_data", "class_data_seg")
    if '_' in class_name:
        class_name = class_name.replace('_', ' ')
    print(class_name)
    if args.seed is not None:
        seed = str(args.seed)
        same_seeds(args.seed)
    else:
        seed = None
    input_args=[
        '--pretrained_model_name_or_path', pretrained_model_name_or_path,
        '--instance_data_dir', instance_data_dir,
        '--instance_seg_dir', instance_seg_dir,
        '--output_dir', output_dir,
        '--instance_prompt', "a sks {}".format(class_name),
        '--resolution', '512',
        '--prior_loss_weight', str(config['prior_loss_weight']),
        '--num_class_images', str(args.num_class_images),
        '--mixed_precision', args.mixed_precision,
        '--prior_generation_precision', args.prior_generation_precision,
        '--class_data_dir', class_data_dir,
        '--class_seg_dir', class_seg_dir,
        '--class_prompt', "a {}".format(class_name),
        '--train_batch_size', '2',
        '--gradient_accumulation_steps', '1',
        '--learning_rate', str(config['lr']),
        '--lr_scheduler', 'constant',
        '--lr_warmup_steps', '0',
        '--max_train_steps', str(args.max_train_steps),
        '--checkpointing_steps', '200',
        '--mse_weight', str(args.mse_weight),
        '--lora_r', str(int(config['lora_r'])),
        '--lora_alpha', str(int(config['lora_alpha'])),
        # '--train_text_encoder', str(args.train_text_encoder),
    ]
    # print('1 >>>>', input_args)
    if args.train_text_encoder:
        input_args += ['--train_text_encoder']
    if not args.single_run:
        input_args += ['--disable_progress_bar']
    if args.seed is not None:
        input_args += ['--seed', seed]
    if args.single_run_push_to_hub:
        input_args += '--push_to_hub'
    if args.with_prior_preservation:
        input_args += ['--with_prior_preservation']
    if args.with_attention_reg:
        input_args += ['--with_attention_reg']
        input_args += ['--attention_loss_weight', str(config['attention_loss_weight'])]
        input_args += ['--attention_reg_token_idx1', str(args.attention_reg_token_idx1)]
        input_args += ['--attention_reg_token_idx2', str(args.attention_reg_token_idx2)]
        input_args += ['--weight_8', '0.3']
        input_args += ['--weight_16', '0.5']
        input_args += ['--weight_32', '0.1']
        input_args += ['--weight_64', '0.1']
        if args.with_attention_reg_snr:
            input_args += ['--with_attention_reg_snr', '--reg_snr_gamma', str(config['reg_snr_gamma'])]
        if args.with_attention_reg_sigmoid:
            input_args += ['--with_attention_reg_sigmoid', '--reg_sigmoid_delta', str(config['reg_sigmoid_delta']), '--reg_sigmoid_kappa', str(config['reg_sigmoid_kappa'])]
        if args.disable_input_reg:
            input_args += ['--disable_input_reg']
        input_args += [
            '--self_attn',
            '--alpha', str(config['alpha']),
            '--beta', str(config['beta']),
            '--reg_snr_mode', str(args.reg_snr_mode),
        ]
    # print('2 >>>>', input_args)
    train_args = get_args(input_args)

    if args.lora:
        lora_train(train_args)
    else:
        scratch_train(train_args)

    save_root = output_dir.split('/')[0]
    generated_images_dir = generate_images_for_evaluation(
            'sks',
            class_name,
            args.max_train_steps,
            output_dir,
            live_subject=CLASS_MAP[class_name],
            num_per_prompt=args.eval_num_per_prompt,
            lora=args.lora,
            finetune_model=args.pretrained_model_name_or_path,
            save_dir=output_dir.replace(save_root, '{}/generated_images'.format(save_root))
    )
    
    fid_score = fid_evaluation(instance_data_dir, generated_images_dir)
    print(">>> FID evaluation done")
    clip_i_score = clip_i_evaluation(instance_data_dir, generated_images_dir)
    print(">>> CLIP-I evaluation done")
    clip_t_score = clip_t_evaluation(generated_images_dir)
    print(">>> CLIP-T evaluation done")
    dino_score = dino_evaluation(instance_data_dir, generated_images_dir)
    print(">>> DINO evaluation done")
    print(">>>>> DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}".format(dino_score, clip_i_score, clip_t_score, fid_score))
    overall = dino_score * args.DINO_weight + clip_i_score * args.CLIP_I_weight + clip_t_score * args.CLIP_T_weight - fid_score * args.FID_weight
    print(">>>>> Overall: {:.4f}".format(overall))
    train.report({
        "overall_performance": overall,
        "DINO": dino_score,
        "CLIP-I": clip_i_score,
        "CLIP-T": clip_t_score,
        "FID": fid_score,
    })


if __name__ == "__main__":
    class_name = "dog"
    pretrained_model_name_or_path = "/dev/shm/alexJiang/source/SD1_4"
    selected = select_reg_semantics(pretrained_model_name_or_path, class_name, select=3)
    print(">>>> selected: ", selected)

    