from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose

import torchvision.transforms.v2 as transforms
import torch

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


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_seg_root, ### segmentation root
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_seg_root=None, ### class data segmentation root
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        augmentation=None,
        single_class_token=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.augmentation = augmentation
        self.single_class_token = single_class_token

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root: {self.instance_data_root} doesn't exists.")
        self.instance_seg_root = Path(instance_seg_root)
        if not self.instance_seg_root.exists():
            raise ValueError(f"Instance {self.instance_seg_root} segmentations root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())

        self.instance_segs_path = []
        for p in self.instance_images_path:
          p_ = str(p)
          p_ = p_.replace(str(self.instance_data_root), str(self.instance_seg_root))
          self.instance_segs_path.append(Path(p_))
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_seg_root = Path(class_seg_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            # self.class_seg_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            # self.class_seg_path = list(self.class_seg_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.segmentation_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        instance_seg = Image.open(self.instance_segs_path[index % self.num_instance_images])
        instance_seg = exif_transpose(instance_seg) # mode 'L': 8-bit grayscale
        if self.augmentation is not None:
            instance_image, instance_seg, aug_str = self.augmentation(instance_image, instance_seg)
            # print(aug_str)
            # display(instance_image)
            # display(instance_seg)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        instance_seg = self.segmentation_transforms(instance_seg)
        instance_seg = torch.where(instance_seg < 0.5, 0, 1).float()
        example["instance_segmentations"] = instance_seg # [1, 768, 768]

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            img_path = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(img_path)
            class_image_seg_path = str(img_path).replace("class_data", "class_data_seg")
            if not self.single_class_token:
                # print(class_image_seg_path)
                # print(class_image_seg_path.split("/")[-1])
                # print(class_image_seg_path.split("/")[-1].split(".")[0])
                class_token = class_image_seg_path.split("/")[-1].split(".")[0].split("_")[-1]
                # print("class_token 1", class_token)
                tokens = class_token.split('-')
                key_idx = len(tokens) + 1
                class_token = class_token.replace('-', ' ')
                # print("class_token 2", class_token)
            else:
                key_idx = len(self.class_prompt.split(' '))

            class_image_seg = Image.open(class_image_seg_path)
            class_image = exif_transpose(class_image)
            class_image_seg = exif_transpose(class_image_seg)

            class_image_seg = self.segmentation_transforms(class_image_seg)
            class_image_seg = torch.where(class_image_seg < 0.5, 0, 1).float()

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_image_segmentations"] = class_image_seg

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                if self.single_class_token:
                    class_text_inputs = tokenize_prompt(
                        self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                    )
                else:
                    class_text_inputs = tokenize_prompt(
                        self.tokenizer, f"a {class_token}", tokenizer_max_length=self.tokenizer_max_length
                    )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask
                example["class_key_idx"] = key_idx
                example["class_prompt"] = self.class_prompt if self.single_class_token else f"a {class_token}"

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    seg_values = [example["instance_segmentations"] for example in examples] ###
    key_idx = [] # for debugging
    class_prompts = [] # for debugging

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        seg_values += [example["class_image_segmentations"] for example in examples]
        key_idx += [example["class_key_idx"] for example in examples]
        class_prompts += [example["class_prompt"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    seg_values = torch.stack(seg_values) ###
    seg_values = seg_values.to(memory_format=torch.contiguous_format) ###
    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "seg_values": seg_values,
        "key_idx": key_idx,
        "class_prompts": class_prompts
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch