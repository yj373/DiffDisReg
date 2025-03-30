from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
# from huggingface_hub import delete_cached_repo

import torch, os, shutil, argparse
import sys
sys.path.append("./")

from engine.evaluation import fid_evaluation, clip_i_evaluation, clip_t_evaluation, dino_evaluation


def generate_images_for_evaluation(
        unique_token,
        class_token,
        step,
        root,
        live_subject=True,
        num_per_prompt=4,
        lora=True,
        finetune_model="runwayml/stable-diffusion-v1-5",
        save_dir="generated_images",
        idx='',
        forgetting_test=False,
):  
    print(save_dir)
    if live_subject:
        ## live subject prompt
        if len(unique_token) > 0:
            prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            'a {0} {1} on the beach'.format(unique_token, class_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            'a {0} {1} with a city in the background'.format(unique_token, class_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            'a {0} {1} wearing a red hat'.format(unique_token, class_token),
            'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
            'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
            'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
            'a {0} {1} in a chef outfit'.format(unique_token, class_token),
            'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
            'a {0} {1} in a police outfit'.format(unique_token, class_token),
            'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
            'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
            'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
            'a red {0} {1}'.format(unique_token, class_token),
            'a purple {0} {1}'.format(unique_token, class_token),
            'a shiny {0} {1}'.format(unique_token, class_token),
            'a wet {0} {1}'.format(unique_token, class_token),
            'a cube shaped {0} {1}'.format(unique_token, class_token)
            ]
        else:
            prompt_list = [
            'a {0} in the jungle'.format(class_token),
            'a {0} in the snow'.format(class_token),
            'a {0} on the beach'.format(class_token),
            'a {0} on a cobblestone street'.format(class_token),
            'a {0} on top of pink fabric'.format(class_token),
            'a {0} on top of a wooden floor'.format(class_token),
            'a {0} with a city in the background'.format(class_token),
            'a {0} with a mountain in the background'.format(class_token),
            'a {0} with a blue house in the background'.format(class_token),
            'a {0} on top of a purple rug in a forest'.format(class_token),
            'a {0} wearing a red hat'.format(class_token),
            'a {0} wearing a santa hat'.format(class_token),
            'a {0} wearing a rainbow scarf'.format(class_token),
            'a {0} wearing a black top hat and a monocle'.format(class_token),
            'a {0} in a chef outfit'.format(class_token),
            'a {0} in a firefighter outfit'.format(class_token),
            'a {0} in a police outfit'.format(class_token),
            'a {0} wearing pink glasses'.format(class_token),
            'a {0} wearing a yellow shirt'.format(class_token),
            'a {0} in a purple wizard outfit'.format(class_token),
            'a red {0} '.format(class_token),
            'a purple {0}'.format(class_token),
            'a shiny {0}'.format(class_token),
            'a wet {0}'.format(class_token),
            'a cube shaped {0}'.format(class_token)
            ]

    else:
        ## object prompt
        if len(unique_token) > 0:
            prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            'a {0} {1} on the beach'.format(unique_token, class_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            'a {0} {1} with a city in the background'.format(unique_token, class_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
            'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
            'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
            'a {0} {1} floating on top of water'.format(unique_token, class_token),
            'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
            'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
            'a {0} {1} on top of a mirror'.format(unique_token, class_token),
            'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
            'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
            'a {0} {1} on top of a white rug'.format(unique_token, class_token),
            'a red {0} {1}'.format(unique_token, class_token),
            'a purple {0} {1}'.format(unique_token, class_token),
            'a shiny {0} {1}'.format(unique_token, class_token),
            'a wet {0} {1}'.format(unique_token, class_token),
            'a cube shaped {0} {1}'.format(unique_token, class_token)
            ]
        else:
            prompt_list = [
            'a {0} in the jungle'.format(class_token),
            'a {0} in the snow'.format(class_token),
            'a {0} on the beach'.format(class_token),
            'a {0} on a cobblestone street'.format(class_token),
            'a {0} on top of pink fabric'.format(class_token),
            'a {0} on top of a wooden floor'.format(class_token),
            'a {0} with a city in the background'.format(class_token),
            'a {0} with a mountain in the background'.format(class_token),
            'a {0} with a blue house in the background'.format(class_token),
            'a {0} on top of a purple rug in a forest'.format(class_token),
            'a {0} with a wheat field in the background'.format(class_token),
            'a {0} with a tree and autumn leaves in the background'.format(class_token),
            'a {0} with the Eiffel Tower in the background'.format(class_token),
            'a {0} floating on top of water'.format(class_token),
            'a {0} floating in an ocean of milk'.format(class_token),
            'a {0} on top of green grass with sunflowers around it'.format(class_token),
            'a {0} on top of a mirror'.format(class_token),
            'a {0} on top of the sidewalk in a crowded street'.format(class_token),
            'a {0} on top of a dirt road'.format(class_token),
            'a {0} on top of a white rug'.format(class_token),
            'a red {0}'.format(class_token),
            'a purple {0}'.format(class_token),
            'a shiny {0}'.format(class_token),
            'a wet {0}'.format(class_token),
            'a cube shaped {0}'.format(class_token)
            ]


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if len(idx) > 0:
        subfolder_name = class_token + idx
    else:
        subfolder_name = class_token
    save_dir_ = os.path.join(save_dir, 'checkpoint-{}'.format(step), '{}-{}'.format(unique_token, subfolder_name))
    if os.path.isdir(save_dir_):
        if not forgetting_test or len(os.listdir(save_dir_)) != num_per_prompt * len(prompt_list):
            shutil.rmtree(save_dir_)
        else:
            print(f">>> No need to regenerate images, for {save_dir_} exits")
            return save_dir_
    os.makedirs(save_dir_)
    if lora and step > 0:
        # print('>>> loading pipeline')
        pipeline = DiffusionPipeline.from_pretrained(
            finetune_model, dtype=torch.float32, safety_checker=None,
        ).to("cuda:0")
        # print('>>> loading lora weight')
        if step > 0 :
            pipeline.load_lora_weights(root, subfolder="checkpoint-{}".format(step), weight_name="pytorch_lora_weights.safetensors")
        else:
            pipeline.load_lora_weights(root, weight_name="pytorch_lora_weights.safetensors")
        pipeline.set_progress_bar_config(disable=True)
        # print('>>>')
    else:
        if step > 0 :
            unet = UNet2DConditionModel.from_pretrained(root, subfolder="checkpoint-{}/unet".format(step))
            text_encoder = CLIPTextModel.from_pretrained(root, subfolder="text_encoder")
            pipeline = DiffusionPipeline.from_pretrained(
                finetune_model, unet=unet, text_encoder=text_encoder, dtype=torch.float32, safety_checker=None
            ).to("cuda:0")
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                finetune_model, dtype=torch.float32, safety_checker=None
            ).to("cuda:0")
        pipeline.set_progress_bar_config(disable=True)
    
    for i, prompt in enumerate(prompt_list):
        for j in range(num_per_prompt):
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            img_name = '{}-{}-{}-{}.png'.format(unique_token, class_token, i, j)
            # print('>>> save: ', img_name)
            img_path = os.path.join(save_dir_, img_name)
            image.save(img_path)
            
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir_


def eval_once(step, root, unique_token, class_token, save_dir, idx, args):
    if not args.disable_generation:
        generated_images_dir = generate_images_for_evaluation(
            unique_token,
            class_token,
            step,
            root,
            args.live_subject,
            args.num_per_prompt,
            args.lora,
            args.finetune_model,
            save_dir,
            idx,
        )
    else:
        assert os.path.isdir(args.generated_images_dir), "generated_images_dir does not exist!"
        generated_images_dir = args.generated_images_dir
    if os.path.isdir(args.instance_data_dir):
        fid_score = fid_evaluation(args.instance_data_dir, generated_images_dir)
        print(">>> FID evaluation done")
        clip_i_score = clip_i_evaluation(args.instance_data_dir, generated_images_dir)
        print(">>> CLIP-I evaluation done")
        clip_t_score = clip_t_evaluation(generated_images_dir)
        print(">>> CLIP-T evaluation done")
        dino_score = dino_evaluation(args.instance_data_dir, generated_images_dir)
        print(">>> DINO evaluation done")
        print(">>>>> DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}".format(dino_score, clip_i_score, clip_t_score, fid_score))
        overall = dino_score * args.DINO_weight + clip_i_score * args.CLIP_I_weight + clip_t_score * args.CLIP_T_weight + fid_score * args.FID_weight
        print(">>>>> Overall: {:.4f}".format(overall))
        return (dino_score, clip_i_score, clip_t_score, fid_score)
    else:
        print("instance_data_dir does not exist!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unique_token",
        type=str,
        required=True,
        help="Unique token for the object of interest.",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        required=True,
        help="Class token for the object of interest.",
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
        "--idx",
        type=str,
        default="",
        help="idx of subfolder to save the generated images.",
    )
    parser.add_argument(
        "--live_subject",
        action="store_true",
        help="Flag to indicate if the object of interest is a live subject.",
    )
    parser.add_argument(
        "--num_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt.",
    )
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
        "--save_dir",
        type=str,
        default="generated_images",
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="generated_images",
        help="Directory to real images.",
    )
    parser.add_argument(
        "--generated_images_dir",
        type=str,
        default="",
        help="Directory to generated images.",
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
        default=-0.001,
        help="Weight of FID in the overall metric",
    )
    parser.add_argument(
        "--disable_generation",
        action="store_true",
        help="Flag to disable image generation",
    )
    args = parser.parse_args()
    if not args.disable_generation:
        generated_images_dir = generate_images_for_evaluation(
            args.unique_token,
            args.class_token,
            args.step,
            args.root,
            args.live_subject,
            args.num_per_prompt,
            args.lora,
            args.finetune_model,
            args.save_dir,
            args.idx,
            forgetting_test=True,
        )
    else:
        assert os.path.isdir(args.generated_images_dir), "generated_images_dir does not exist!"
        generated_images_dir = args.generated_images_dir
    if os.path.isdir(args.instance_data_dir):
        fid_score = fid_evaluation(args.instance_data_dir, generated_images_dir)
        print(">>> FID evaluation done")
        clip_i_score = clip_i_evaluation(args.instance_data_dir, generated_images_dir)
        print(">>> CLIP-I evaluation done")
        clip_t_score = clip_t_evaluation(generated_images_dir)
        print(">>> CLIP-T evaluation done")
        dino_score = dino_evaluation(args.instance_data_dir, generated_images_dir)
        print(">>> DINO evaluation done")
        print(">>>>> DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}".format(dino_score, clip_i_score, clip_t_score, fid_score))
        overall = dino_score * args.DINO_weight + clip_i_score * args.CLIP_I_weight + clip_t_score * args.CLIP_T_weight - fid_score * args.FID_weight
        print(">>>>> Overall: {:.4f}".format(overall))
    else:
        print("instance_data_dir {} does not exist!".format(args.instance_data_dir))