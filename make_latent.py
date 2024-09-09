import os

HOME_DIR = os.environ.get('HOME', '/root')

import sys

sys.path.append(HOME_DIR + '/GITHUB/aravindhv10/x-flux')

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file
from safetensors.torch import load_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip, load_flow_model2,
                           load_t5)
from src.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor

from image_datasets.dataset import loader, loader_mine

from prodigyopt import Prodigy

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def load_image(path):
    img = Image.open(path)
    img = c_crop(img)
    img = img.resize((1024, 1024))
    img = torch.from_numpy((np.array(img) / 127.5) - 1)
    img = img.permute(2, 0, 1)
    return img[None, :]


def load_vae(name='flux-dev'):
    vae = load_ae(name, device="cuda").to(dtype=torch.float16)
    return vae


def get_embedding(path_image, auto_encoder):
    image = load_image(path=path_image)
    return auto_encoder.encode(image.to('cuda').to(torch.float16)).squeeze(0)


def remove_extension(path_input):
    loc = path_input.rfind('.')
    return path_input[0:loc]


class image_to_ae_safetensors:

    def __init__(self, name='flux-dev'):
        self.vae = load_vae(name='flux-dev')
        self.vae.requires_grad_(False)

    def process_image(self, path_image, path_output):
        if os.path.exists(path_output):
            return
        embedding = get_embedding(path_image=path_image, auto_encoder=self.vae)
        out = {'encoded_image': embedding}
        save_file(out, path_output)


class text_embedders:

    def __init__(self, is_schnell=False):

        self.t5 = load_t5(
            'cuda',
            max_length=256 if is_schnell else 512).to(dtype=torch.float16)

        self.clip = load_clip('cuda').to(dtype=torch.float16)

        self.t5.requires_grad_(False)
        self.clip.requires_grad_(False)

    def process_image_prompt(self, ae_sft_path, prompt_path, output_path):
        if os.path.exists(output_path):
            return

        ae_latent = load_file(ae_sft_path)['encoded_image'].unsqueeze(0)
        prompt = open(prompt_path, 'r', encoding='utf-8').read()

        inp = prepare(t5=self.t5, clip=self.clip, img=ae_latent, prompt=prompt)

        for i in inp.keys():
            inp[i] = inp[i].squeeze(0).to(dtype=torch.float16)

        save_file(inp, output_path)


def create_embeddings(list_path_images, list_path_captions):
    list_path_ae_output = list(
        remove_extension(path_input=i) + '_ae.safetensors'
        for i in list_path_images)

    slave = image_to_ae_safetensors()
    for i in range(len(list_path_images)):
        slave.process_image(path_image=list_path_images[i],
                            path_output=list_path_ae_output[i])

    list_path_text_embedding_output = list(
        remove_extension(path_input=i) + '_text.safetensors'
        for i in list_path_images)

    slave = text_embedders()
    for i in range(len(list_path_images)):
        slave.process_image_prompt(
            ae_sft_path=list_path_ae_output[i],
            prompt_path=list_path_captions[i],
            output_path=list_path_text_embedding_output[i])


def create_embeddings_in_docker():

    DIR_INPUT = '/data/input'
    list_images = list(
        os.path.join(DIR_INPUT, i) for i in os.listdir(DIR_INPUT)
        if (i.lower().endswith('.png') or i.lower().endswith('.jpg')
            or i.lower().endswith('.jpeg')))

    list_images.sort()

    list_captions = list(
        remove_extension(path_input=i) + '.txt' for i in list_images)

    create_embeddings(list_path_images=list_images,
                      list_path_captions=list_captions)


def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main():
    create_embeddings_in_docker()


if __name__ == "__main__":
    main()
