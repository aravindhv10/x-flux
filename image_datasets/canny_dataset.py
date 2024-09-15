import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2


def remove_extension(path_input):
    loc = path_input.rfind('.')
    return path_input[0:loc]


def get_main_image_files(path_dir_input):
    list_path_dir = []

    for (root, dirs, files) in os.walk(path_dir_input):
        for file in files:
            tmp = file.lower()
            if (tmp == 'image.png') or (tmp
                                        == 'image.jpg') or (tmp
                                                            == 'image.jpeg'):
                list_path_dir.append(os.path.join(root, file))

    return list_path_dir


def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def c_crop_2(image):
    width, height = image.size
    new_size = max(width, height)
    # left = (width - new_size) / 2
    # top = (height - new_size) / 2
    # right = (width + new_size) / 2
    # bottom = (height + new_size) / 2

    left = 0
    top = 0
    right = new_size
    bottom = new_size
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
            hint = canny_processor(img)
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.json'
            prompt = json.load(open(json_path))['caption']
            return img, hint, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class CustomImageDataset_cached(Dataset):

    def __init__(self, img_dir='/data/input', img_size=1024):

        self.img_size = img_size

        list_path_image = get_main_image_files(path_dir_input=img_dir)
        list_path_image.sort()

        self.list_path_control = list(
            os.path.dirname(i) + '/control.png' for i in list_path_image)

        self.list_path_caption = list(
            os.path.dirname(i) + '/caption.safetensors'
            for i in list_path_image)

        self.list_path_image = list(
            os.path.dirname(i) + '/image.safetensors' for i in list_path_image)

    def __len__(self):
        return len(self.list_path_image)

    def __getitem__(self, idx):
        try:
            hint = Image.open(self.list_path_control[idx])
            hint = c_crop_2(hint)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            ae = load_file(self.list_path_image[idx])['encoded_image']
            emb = load_file(self.list_path_caption[idx])

            return ae, hint, emb
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset,
                      batch_size=train_batch_size,
                      num_workers=num_workers,
                      shuffle=True)


def loader_2(train_batch_size, num_workers, **args):

    dataset = CustomImageDataset_cached(**args)
    return DataLoader(dataset,
                      batch_size=train_batch_size,
                      num_workers=num_workers,
                      shuffle=True)
