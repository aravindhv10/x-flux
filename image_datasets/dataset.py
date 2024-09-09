import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from safetensors.torch import load_file


def remove_extension(path_input):
    loc = path_input.rfind('.')
    return path_input[0:loc]


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, img_size=512):
        self.images = [
            os.path.join(img_dir, i) for i in os.listdir(img_dir)
            if '.jpg' in i or '.png' in i
        ]
        self.images.sort()
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx])
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.json'
            prompt = json.load(open(json_path))['caption']
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class CachedDataset(Dataset):

    def __init__(
        self,
        DIR_INPUT='/data/input',
        img_size=1024,
    ):

        self.list_path_images = list(
            os.path.join(DIR_INPUT, i) for i in os.listdir(DIR_INPUT)
            if (i.lower().endswith('.png') or i.lower().endswith('.jpg')
                or i.lower().endswith('.jpeg')))

        self.list_path_images.sort()

        self.list_path_captions = list(
            remove_extension(path_input=i) + '.txt'
            for i in self.list_path_images)

        self.list_path_ae_output = list(
            remove_extension(path_input=i) + '_ae.safetensors'
            for i in self.list_path_images)

        self.list_path_text_embedding_output = list(
            remove_extension(path_input=i) + '_text.safetensors'
            for i in self.list_path_images)

        self.img_size = img_size

    def __len__(self):
        return len(self.list_path_images)

    def __getitem__(self, idx):
        ae = load_file(self.list_path_ae_output[idx])['encoded_image']
        emb = load_file(self.list_path_text_embedding_output[idx])
        emb["img"] = emb["img"].squeeze(0)
        emb["img_ids"] = emb["img_ids"].squeeze(0)
        emb["txt"] = emb["txt"].squeeze(0)
        emb["txt_ids"] = emb["txt_ids"].squeeze(0)
        emb["vec"] = emb["vec"].squeeze(0)
        return ae, emb


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset,
                      batch_size=train_batch_size,
                      num_workers=num_workers,
                      shuffle=True)


def loader_mine(train_batch_size, num_workers, **args):
    dataset = CachedDataset()
    return DataLoader(dataset,
                      batch_size=train_batch_size,
                      num_workers=num_workers,
                      shuffle=True)
