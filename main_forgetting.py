import torch, os, shutil, argparse
import numpy as np
from functions.evaluate import generate_images_for_evaluation
from engine.evaluation import fid_evaluation, clip_i_evaluation, clip_t_evaluation, dino_evaluation

from transformers import AutoTokenizer
from transformers import CLIPTextModel
from transformers import BertModel, BertTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

LIVE_SBJS = [
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

SBJS = [
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
#
BASELINE_LIVE_SBJS = [
    1.3112,
    1.3704,
    1.2389,
    1.2358,
    1.0328,
    1.234,
    1.3241,
    1.2712,
    1.2344,
    1.2512,
    1.2598,
    1.3383,
    1.2256,
    1.257,
    1.3155,
    1.2388,
    1.2656,
    1.2959,
    1.312,
    1.1913,
    1.2978    
]

BASELINE_SBJS = [
    1.341,
    1.4519,
    1.6214,
    1.2644,
    1.2914,
    1.4719,
    1.4273,
    1.3013,
    1.331,
    1.33,
    1.286,
    1.4218,
    1.2488,
    1.3473,
    1.3955,
    1.2952,
    1.3319,
    1.371,
    1.2175,
    1.26,
    1.3846,
]


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
        # print(">>> CLIP similarity")
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
        # print(">>> BERT single token")
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
        # print(">>> BERT sentence")
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

    # print(f"Cosine similarity ({concept1}, {concept2}): ", sim)
    return sim.item()


def parser_args():
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
        "--cross_validation",
        type=int,
        default=3,
        help="Number of folders for cross validation.",
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
        "--save_dir_baseline",
        type=str,
        default="",
        help="Directory to save the generated images of baseline.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="generated_images",
        help="Directory to images for comparison.",
    )
    parser.add_argument(
        "--generated_images_dir",
        type=str,
        default="",
        help="Directory to generated images.",
    )
    parser.add_argument(
        "--generated_images_dir_baseline",
        type=str,
        default="",
        help="Directory to generated images.",
    )
    # parser.add_argument(
    #     "--semantic_mode",
    #     type=int,
    #     default=0,
    #     help="how to compute semantic similarity",
    # )
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
    # parser.add_argument(
    #     "--disable_generation",
    #     action="store_true",
    #     help="Flag to disable image generation",
    # )
    parser.add_argument(
        "--disable_baseline_computation",
        action="store_true",
        help="Flag to disable image generation for baseline",
    )
    parser.add_argument(
        "--start_obj_idx",
        type=int,
        default=0,
        help="Index of the index to start.",
    )
    parser.add_argument(
        "--end_obj_idx",
        type=int,
        default=21,
        help="Index of the index to end.",
    )
    args = parser.parse_args()
    
    return args


def main(args):
    if args.live_subject:
        concepts = LIVE_SBJS
    else:
        concepts = SBJS

    tokenizer = AutoTokenizer.from_pretrained(
        args.finetune_model,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(args.finetune_model, subfolder="text_encoder")

    for idx, c in enumerate(concepts[max(args.start_obj_idx, 0):min(args.end_obj_idx, len(concepts))]):
        print(f">>> testing concept: {c}")
        
        sims = []
        for mode in range(3):
            sims.append(compute_semantic_sim(tokenizer, text_encoder, args.class_token, c, mode=mode, live_subject=args.live_subject))
        if not args.disable_baseline_computation:
            baseline_images_dirs = []
            for i in range(args.cross_validation):
                dir = generate_images_for_evaluation(
                    "",
                    c,
                    0,
                    "",
                    live_subject=args.live_subject,
                    num_per_prompt=args.num_per_prompt,
                    lora=args.lora,
                    finetune_model=args.finetune_model,
                    save_dir=args.save_dir_baseline,
                    idx= "" if i == 0 else str(i),
                    forgetting_test=True,
                )
                baseline_images_dirs.append(dir)

            fid_base = []
            clip_i_base = []
            clip_t_base = []
            dino_base = []
            overall_base = []
            for i in range(args.cross_validation):
                instance_dir = baseline_images_dirs[i]
                clip_t_score = clip_t_evaluation(instance_dir)
                clip_t_base.append(clip_t_score)
                for j in range(i + 1, args.cross_validation):
                    generated_dir = baseline_images_dirs[j]
                    fid_score = fid_evaluation(instance_dir, generated_dir)
                    fid_base.append(fid_score)
                    clip_i_score = clip_i_evaluation(instance_dir, generated_dir)
                    clip_i_base.append(clip_i_score)
                    dino_score = dino_evaluation(instance_dir, generated_dir)
                    dino_base.append(dino_score)
                    overall = dino_score * args.DINO_weight + clip_i_score * args.CLIP_I_weight + clip_t_score * args.CLIP_T_weight + fid_score * args.FID_weight
                    overall_base.append(overall)
                    print(">>>>> ({}, {}) DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}, Overall: {:.4f}".format(i, j, dino_score, clip_i_score, clip_t_score, fid_score, overall))

            fid_base = np.array(fid_base).mean()
            clip_i_base = np.array(clip_i_base).mean()
            clip_t_base = np.array(clip_t_base).mean()
            dino_base = np.array(dino_base).mean()
            overall_base = np.array(overall_base).mean()
            print(">>>>> DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}, Overall: {:.4f}".format(dino_base, clip_i_base, clip_t_base, fid_base, overall_base))

            all_baseline_images_dir = baseline_images_dirs[0].replace(c, c + "_overall")
            print("all the baseline images: ", all_baseline_images_dir)
            if not os.path.isdir(all_baseline_images_dir):
                os.makedirs(all_baseline_images_dir, exist_ok=True)
                for k, folder in enumerate(baseline_images_dirs):
                    for file_name in os.listdir(folder):
                        full_file_name = os.path.join(folder, file_name)
                        if os.path.isfile(full_file_name):
                            new_file_name = str(k) + '_' + file_name
                            shutil.copy(full_file_name, os.path.join(all_baseline_images_dir, new_file_name))
        else:
            overall_base = BASELINE_LIVE_SBJS[max(0, args.start_obj_idx) + idx] if args.live_subject else BASELINE_SBJS[max(0, args.start_obj_idx) + idx]
            # all_baseline_images_dir = os.path.join(args.save_dir_baseline, 'checkpoint-0', '{}-{}_overall'.format("", c))
            baseline_images_dirs = []
            for k in range(args.cross_validation):
                if k == 0:
                    dir = os.path.join(args.save_dir_baseline, 'checkpoint-0', '{}-{}'.format("", c))
                else:
                    dir = os.path.join(args.save_dir_baseline, 'checkpoint-0', '{}-{}{}'.format("", c, k))
                baseline_images_dirs.append(dir)
            print(">>>>> Precomputed baseline for {}, overall baseline: {:.4f}".format(c, overall_base))
        
        tuned_generated_images_dirs = []
        for i in range(args.cross_validation):
            dir = generate_images_for_evaluation(
                "",
                c,
                args.step,
                args.root,
                live_subject=args.live_subject,
                num_per_prompt=args.num_per_prompt,
                lora=args.lora,
                finetune_model=args.finetune_model,
                save_dir=args.save_dir,
                idx= "" if i == 0 else str(i),
                forgetting_test=True,
            )
            tuned_generated_images_dirs.append(dir)
            print("tuned generation:", dir)

        fid_tuned = []
        clip_i_tuned = []
        clip_t_tuned = []
        dino_tuned = []
        overall_tuned = []

        for i in range(args.cross_validation):
            generated_dir = tuned_generated_images_dirs[i]
            clip_t_score = clip_t_evaluation(generated_dir)
            clip_t_tuned.append(clip_t_score)
            # fid_score = fid_evaluation(all_baseline_images_dir, generated_dir)
            fid_score = fid_evaluation(baseline_images_dirs[i], generated_dir)
            fid_tuned.append(fid_score)
            clip_i_score = clip_i_evaluation(baseline_images_dirs[i], generated_dir)
            clip_i_tuned.append(clip_i_score)
            dino_score = dino_evaluation(baseline_images_dirs[i], generated_dir)
            dino_tuned.append(dino_score)
            overall = dino_score * args.DINO_weight + clip_i_score * args.CLIP_I_weight + clip_t_score * args.CLIP_T_weight + fid_score * args.FID_weight
            overall_tuned.append(overall)
            print(">>>>> ({}) DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}, Overall: {:.4f}".format(i, dino_score, clip_i_score, clip_t_score, fid_score, overall))

        fid_tuned = np.array(fid_tuned).mean()
        clip_i_tuned = np.array(clip_i_tuned).mean()
        clip_t_tuned = np.array(clip_t_tuned).mean()
        dino_tuned = np.array(dino_tuned).mean()
        overall_tuned = np.array(overall_tuned).mean()
        # print(">>>>> DINO score: {:.4f}, CLIP-I score:  {:.4f}, CLIP-T score: {:.4f}, FID score: {:.4f}, Overall: {:.4f}".format(dino_base, clip_i_base, clip_t_base, fid_base, overall_base))

        deg = (overall_tuned - overall_base) * 100.0 / overall_base
        print(">>>>> Overall degradation: {:.2f}".format(deg))
        for i, sim in enumerate(sims):
            print(">>> ({}, {}), similarity {:.4f}, mode: {}".format(args.class_token, c, sim, i))


if __name__ == "__main__":
    args = parser_args()
    main(args)
    print(">>> Done!")



    


