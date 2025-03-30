from transformers import ViTImageProcessor, ViTModel, AutoProcessor, CLIPVisionModel
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchmetrics.multimodal import CLIPScore
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3

import torch.nn.functional as F
import os, torch
import numpy as np
import pathlib
from tqdm import tqdm
from scipy import linalg


### DINO score
def extract_features_dino(model, processor, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        features = outputs.last_hidden_state
    return features.flatten().unsqueeze(0)


def compute_dino_score(real_image, generated_image, model, processor):
    real_features = extract_features_dino(model, processor, real_image)
    generated_features = extract_features_dino(model, processor, generated_image)
    similarity = F.cosine_similarity(real_features, generated_features)
    return similarity.item()


def dino_evaluation(real_image_dir, generated_image_dir, generated_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained('/dev/shm/alexJiang/source/DINO')
    model = ViTModel.from_pretrained('/dev/shm/alexJiang/source/DINO').to(device)
    dino_scores = []
    for real_image_name in os.listdir(real_image_dir):
        real_image_path = os.path.join(real_image_dir, real_image_name)
        real_image = Image.open(real_image_path)
        for generated_image_name in os.listdir(generated_image_dir):
            if generated_id is not None and not generated_image_name.endswith(generated_id):
                continue
            generated_image_path = os.path.join(generated_image_dir, generated_image_name)
            generated_image = Image.open(generated_image_path)
            dino_scores.append(compute_dino_score(real_image, generated_image, model, processor))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sum(dino_scores) / len(dino_scores)


### CLIP-T score
def clip_t_evaluation(generated_image_dir, live_subject=False, class_token = 'backpack'):
    if live_subject:
        ## live subject prompt
        prompt_list = [
            'a {} in the jungle'.format(class_token),
            'a {} in the snow'.format(class_token),
            'a {} on the beach'.format(class_token),
            'a {} on a cobblestone street'.format(class_token),
            'a {} on top of pink fabric'.format(class_token),
            'a {} on top of a wooden floor'.format(class_token),
            'a {} with a city in the background'.format(class_token),
            'a {} with a mountain in the background'.format(class_token),
            'a {} with a blue house in the background'.format(class_token),
            'a {} on top of a purple rug in a forest'.format(class_token),
            'a {} wearing a red hat'.format(class_token),
            'a {} wearing a santa hat'.format(class_token),
            'a {} wearing a rainbow scarf'.format(class_token),
            'a {} wearing a black top hat and a monocle'.format(class_token),
            'a {} in a chef outfit'.format(class_token),
            'a {} in a firefighter outfit'.format(class_token),
            'a {} in a police outfit'.format(class_token),
            'a {} wearing pink glasses'.format(class_token),
            'a {} wearing a yellow shirt'.format(class_token),
            'a {} in a purple wizard outfit'.format(class_token),
            'a red {}'.format(class_token),
            'a purple {}'.format(class_token),
            'a shiny {}'.format(class_token),
            'a wet {}'.format(class_token),
            'a cube shaped {}'.format(class_token)
        ]
    else:
        ## object prompt
        prompt_list = [
            'a {} in the jungle'.format(class_token),
            'a {} in the snow'.format(class_token),
            'a {} on the beach'.format(class_token),
            'a {} on a cobblestone street'.format(class_token),
            'a {} on top of pink fabric'.format(class_token),
            'a {} on top of a wooden floor'.format(class_token),
            'a {} with a city in the background'.format(class_token),
            'a {} with a mountain in the background'.format(class_token),
            'a {} with a blue house in the background'.format(class_token),
            'a {} on top of a purple rug in a forest'.format(class_token),
            'a {} with a wheat field in the background'.format(class_token),
            'a {} with a tree and autumn leaves in the background'.format(class_token),
            'a {} with the Eiffel Tower in the background'.format(class_token),
            'a {} floating on top of water'.format(class_token),
            'a {} floating in an ocean of milk'.format(class_token),
            'a {} on top of green grass with sunflowers around it'.format(class_token),
            'a {} on top of a mirror'.format(class_token),
            'a {} on top of the sidewalk in a crowded street'.format(class_token),
            'a {} on top of a dirt road'.format(class_token),
            'a {} on top of a white rug'.format(class_token),
            'a red {}'.format(class_token),
            'a purple {}'.format(class_token),
            'a shiny {}'.format(class_token),
            'a wet {}'.format(class_token),
            'a cube shaped {}'.format(class_token)
        ]
    image_paths =[]
    captions = []
    for image in os.listdir(generated_image_dir):
        image_paths.append(os.path.join(generated_image_dir, image))
        idx = int(image.split('-')[2])
        captions.append(prompt_list[idx])

    clip_score = CLIPScore(model_name_or_path="/dev/shm/alexJiang/source/CLIP")
    clip_t_results = []
    for img_path, caption in zip(image_paths, captions):
        img_tensor = torch.tensor(np.array(Image.open(img_path)))
        clip_t_score = clip_score(img_tensor, caption).item()
        clip_t_results.append(clip_t_score)

    return sum(clip_t_results) / len(clip_t_results)


### CLIP-I score
def extract_features_clip(model, processor, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        features = outputs.pooler_output
    return features


def compute_clip_I_score(real_image, generated_image, model, processor):
    real_features = extract_features_clip(model, processor, real_image)
    generated_features = extract_features_clip(model, processor, generated_image)
    similarity = F.cosine_similarity(real_features, generated_features)
    return similarity.item()


def clip_i_evaluation(real_image_dir, generated_image_dir, generated_id=None):
    clip_i_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModel.from_pretrained("/dev/shm/alexJiang/source/CLIP")
    model.to(device)
    processor = AutoProcessor.from_pretrained("/dev/shm/alexJiang/source/CLIP")

    for real_image_name in os.listdir(real_image_dir):
        real_image_path = os.path.join(real_image_dir, real_image_name)
        
        real_image = Image.open(real_image_path)
        for generated_image_name in os.listdir(generated_image_dir):
            if generated_id is not None and not generated_image_name.endswith(generated_id):
                continue
            generated_image_path = os.path.join(generated_image_dir, generated_image_name)
            generated_image = Image.open(generated_image_path)
            clip_i_scores.append(compute_clip_I_score(real_image, generated_image, model, processor))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sum(clip_i_scores) / len(clip_i_scores)


### FID score
IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)
    trans = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    dataset = ImagePathDataset(files, transforms=trans)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        m, s = calculate_activation_statistics(
            files, model, batch_size, dims, device, num_workers
        )

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fid_value


def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Saves FID statistics of one path"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )

    np.savez_compressed(paths[1], mu=m1, sigma=s1)


def fid_evaluation(real_image_dir, generated_image_dir):
    return calculate_fid_given_paths([real_image_dir, generated_image_dir], 50, "cuda", 2048)