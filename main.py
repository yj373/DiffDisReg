import argparse
import os
import shutil

from functions.train import training_function, select_reg_semantics, clean_class_name

from ray import tune
from ray.tune.search.optuna import OptunaSearch


def parser_args():
    parser = argparse.ArgumentParser(description='DiffDisReg')
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_class_name",
        type=str,
        default="backpack",
        required=True,
        help="Object of interest in the DreamBooth dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/LoRA-DreamBooth-backpack",
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default="",
        help="Specified class data directory.",
    )
    parser.add_argument(
        "--class_data_root",
        type=str,
        default="output/class_data/",
        help="Specified root directory storing class data.",
    )
    parser.add_argument(
        "--semantic_mode",
        type=str,
        default=0,
        help="how to compute semantic similarity",
    )
    parser.add_argument(
        "--num_reg_concepts",
        type=int,
        default=3,
        help="number of concepts for regularization if --class_data_dir is auto",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="number of triles for tuning",
    )
    parser.add_argument(
        "--eval_num_per_prompt",
        type=int,
        default=1,
        help="number of images generated for each prompt during evaluation",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="number of fine-tuning steps",
    )
    parser.add_argument(
        "--num_cpu",
        type=int,
        default=4,
        help="number of CPU cores",
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=1,
        help="number of GPUs",
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
        help="Flag to train with LoRA.",
    )
    parser.add_argument(
        "--DINO_weight",
        type=float,
        default=2.0,
        help="Weight of DINO in the overall metric",
    )
    parser.add_argument(
        "--CLIP_I_weight",
        type=float,
        default=0.5,
        help="Weight of CLIP-I in the overall metric",
    )
    parser.add_argument(
        "--CLIP_T_weight",
        type=float,
        default=0.01,
        help="Weight of CLIP-T in the overall metric",
    )
    parser.add_argument(
        "--FID_weight",
        type=float,
        default=0.001,
        help="Weight of FID in the overall metric",
    )
    parser.add_argument(
        "--single_run",
        action='store_true',
        help="whether to only run a single trial",
    )
    parser.add_argument(
        "--single_run_lr",
        type=float,
        default=1e-4,
        help="lr for the single trial"
    )
    parser.add_argument(
        "--single_run_prior_loss_weight",
        type=float,
        default=1.0,
        help="prior loss weight for the single trial"
    )
    parser.add_argument(
        "--single_run_lora_r",
        type=int,
        default=16,
        help="LoRA rank for the single trial"
    )
    parser.add_argument(
        "--single_run_lora_alpha",
        type=float,
        default=32.0,
        help="LoRA alpha for the single trial"
    )
    parser.add_argument(
        "--single_run_push_to_hub",
        action='store_true',
        help="whether to push to hub",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--with_attention_reg",
        default=False,
        action="store_true",
        help="Flag to add attention regularization.",
    )
    parser.add_argument(
        "--with_attention_reg_snr",
        default=False,
        action="store_true",
        help="Flag to add snr weight to attention regularization.",
    )
    parser.add_argument(
        "--with_attention_reg_sigmoid",
        default=False,
        action="store_true",
        help="Flag to add sigmoid weight to attention regularization.",
    )
    parser.add_argument(
        "--disable_prior_loss",
        default=False,
        action="store_true",
        help="Flag to disable prior reconstruction loss.",
    )
    parser.add_argument(
        "--disable_input_reg",
        default=False,
        action="store_true",
        help="Flag to disable input BCE loss.",
    )
    parser.add_argument(
        "--single_run_attn_reg_loss_weight",
        type=float,
        default=0.1,
        help="Attention regularization loss weight"
    )
    parser.add_argument(
        "--single_run_attn_reg_snr_gamma",
        type=float,
        default=5.0,
        help="Attention regularization snr gamma"
    )
    parser.add_argument(
        "--single_run_attn_reg_sigmoid_delta",
        type=float,
        default=200.0,
        help="Attention regularization sigmoid delta"
    )
    parser.add_argument(
        "--single_run_attn_reg_sigmoid_kappa",
        type=float,
        default=10.0,
        help="Attention regularization sigmoid kappa"
    )
    parser.add_argument(
        "--reg_snr_mode",
        type=str,
        default="0",
        help="regularization snr mode",
    )
    parser.add_argument(
        "--attention_reg_token_idx1",
        type=int,
        default=1,
        help="index of token for attention regularization"
    )
    parser.add_argument(
        "--attention_reg_token_idx2",
        type=int,
        default=-1,
        help="index of token for attention regularization"
    )
    # parser.add_argument(
    #     "--attention_reg_prior_token_idx", 
    #     type=int, 
    #     default=2, 
    #     help="Parameter for the mask construction."
    # )
    parser.add_argument(
        "--single_run_attn_reg_alpha",
        type=float,
        default=60.0,
        help="Attention regularization map alpha"
    )
    parser.add_argument(
        "--single_run_attn_reg_beta",
        type=float,
        default=0.15,
        help="Attention regularization map beta"
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
    parser.add_argument(
        "--mse_weight", 
        type=float, 
        default=1.0, 
        help="The weight of MSE loss."
    )

    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )

    args = parser.parse_args()
    return args


def make_class_data_dir(class_data_dir, class_seg_dir, total=100, root="output/class_data"):
    class_names = class_data_dir.split("/")[-1].split("_")
    select = int(total / len(class_names))
    for i, class_name in enumerate(class_names):
        origin_class_name = class_name
        tokens = class_name.split('-')
        if len(tokens) > 1:
            class_name = '_'.join(tokens)
        select_dir = os.path.join(root, class_name)
        print(select_dir)
        if not os.path.isdir(select_dir):
            raise ValueError(f"Class data directory {select_dir} does not exist.")
        select_seg_dir = select_dir.replace("class_data", "class_data_seg")
        # print(f"selected from {select_dir} with {len(os.listdir(select_dir))} files, {i * select} to {(i + 1) * select}")
        selected_files = os.listdir(select_dir)[0:select]
        # print(len(selected_files))
        for file in selected_files:
            source = os.path.join(select_dir, file)
            file_name = file.split(".")[0]
            file_type = file.split(".")[1]
            new_file = f"{file_name}_{origin_class_name}.{file_type}"
            destination = os.path.join(class_data_dir, new_file)
            # print(f"Copying {source} to {destination}")
            shutil.copy(source, destination)
            source = os.path.join(select_seg_dir, file)
            destination = os.path.join(class_seg_dir, new_file)
            shutil.copy(source, destination)
            # print(f"Copying {source} to {destination}")
            # print(">>>>>>")
   

def main():
    args = parser_args()
    if len(args.class_data_dir) > 0:
        if args.class_data_dir != "auto":
            class_data_dir = args.class_data_dir
        else:
            regu_semantics = select_reg_semantics(args.pretrained_model_name_or_path, args.instance_class_name, select=args.num_reg_concepts, mode=int(args.semantic_mode))
            print("regu semantics:")
            print(regu_semantics)
            class_name = clean_class_name(args.instance_class_name)
            tokens = class_name.split('_')
            if len(tokens) > 1:
                class_name = '-'.join(tokens)
            folder_name = "_".join(regu_semantics)
            folder_name = class_name + '_' + folder_name
            class_data_dir = os.path.join(args.class_data_root, folder_name)
            args.class_data_dir = class_data_dir
        class_seg_dir = class_data_dir.replace("class_data", "class_data_seg")
        if os.path.exists(class_data_dir) and len(os.listdir(class_data_dir)) != args.num_class_images:
            shutil.rmtree(class_data_dir)
            shutil.rmtree(class_seg_dir)
        if not os.path.exists(class_data_dir):
            os.makedirs(class_data_dir)
            os.makedirs(class_seg_dir)
        if len(os.listdir(class_data_dir)) == 0:
            make_class_data_dir(class_data_dir, class_seg_dir, total=args.num_class_images, root=args.class_data_root)
            print(f">>> Class data directory {class_data_dir} ({len(os.listdir(class_data_dir))}) created.")

    if not args.single_run:
        os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
        prior_loss_weight = tune.uniform(0.0, 0.0) if args.disable_prior_loss else tune.uniform(0.0, 1.0)
        attention_loss_weight = tune.uniform(0.0, 1.0) if args.with_attention_reg else tune.uniform(0.0, 0.0)
        learning_rate = tune.loguniform(1e-4, 1e-4) if args.lora else tune.loguniform(5e-6, 5e-6)
        search_space = {
            'lr': learning_rate,
            'prior_loss_weight': prior_loss_weight,
            'lora_r': tune.uniform(32.0, 32.0),
            'lora_alpha': tune.uniform(32.0, 32.0),
            'alpha': tune.uniform(8.0, 8.0),
            'beta': tune.uniform(0.4, 0.4),
            # 'reg_snr_gamma': tune.uniform(1.0, 1.0), # reg snr
            'reg_sigmoid_delta': tune.uniform(300.0, 500.0),
            'reg_sigmoid_kappa': tune.uniform(50.0, 80.0),
            'attention_loss_weight': attention_loss_weight,
        }
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(training_function, args=args), resources={"cpu": args.num_cpu, "gpu": args.num_gpu}),
            param_space=search_space,
            tune_config=tune.TuneConfig(search_alg=OptunaSearch(), metric="overall_performance", mode="max", num_samples=args.num_runs),
        )
        results = tuner.fit()
        print(">>>>>>>>>> Best config is:", results.get_best_result().config)
        best_result = results.get_best_result("overall_performance", "max")
        print(">>> best config: ", best_result.config)
        print(">>> overall performance {:.4f}".format(best_result.metrics["overall_performance"]))
        print(">>> DINO score: {:.4f}, CLIP-I score: {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}".format(best_result.metrics["DINO"], best_result.metrics["CLIP-I"], best_result.metrics["CLIP-T"], best_result.metrics["FID"]))
    else:
        config = {}
        config['lr'] = args.single_run_lr
        config['prior_loss_weight'] = args.single_run_prior_loss_weight
        config['lora_r'] = args.single_run_lora_r
        config['lora_alpha'] = args.single_run_lora_alpha
        if args.with_attention_reg:
            config['alpha'] = args.single_run_attn_reg_alpha
            config['beta'] = args.single_run_attn_reg_beta
            if args.with_attention_reg_snr:
                config['reg_snr_gamma'] = args.single_run_attn_reg_snr_gamma
            if args.with_attention_reg_sigmoid:
                config['reg_sigmoid_delta'] = args.single_run_attn_reg_sigmoid_delta
                config['reg_sigmoid_kappa'] = args.single_run_attn_reg_sigmoid_kappa
            config['attention_loss_weight'] = args.single_run_attn_reg_loss_weight
        print('config', config)
        print('args before training_function', args)
        training_function(config, args)


if __name__ == '__main__':
    main()