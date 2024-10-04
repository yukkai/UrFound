"""
Methods for image and text loading, pre-processing and generation
for vision-language pretraining. Also, it includes data augmentation
utilities.
"""

import numpy as np
import random
import torch
import copy

from PIL import Image
from torchvision.transforms import Resize
from util.flair_dataloader.dictionary import definitions
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


class LoadImage():
    def __init__(self, target="image_path"):
        self.target = target
        """
        Load, organize channels, and standardize intensity of images.
        """

    def __call__(self, data):
        img = Image.open(data[self.target]).convert('RGB')
        data[self.target.replace("_path", "")] = img
        return data


class ImageScaling():

    """
    Method for image scaling. It includes two options: scaling from canvas, to avoid image distortions,
    and regular scaling trough resizing.
    """

    def __init__(self, size=(512, 512), canvas=True, target="image"):
        self.size = size
        self.canvas = canvas
        self.target = target

        # self.transforms = torch.nn.Sequential(
        #     Resize(self.size),
        # )
        self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(448, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                # transforms.Normalize(mean=[0.4978], std=[0.2449])])

    def __call__(self, data):
        img = data[self.target]
        img = self.transforms(img)
        # if not self.canvas or (img.shape[-1] == img.shape[-2]):
        #     img = self.transforms(img)
        # else:
        #     sizes = img.shape[-2:]
        #     max_size = max(sizes)
        #     scale = max_size/self.size[0]
        #     img = Resize((int(img.shape[-2]/scale), int((img.shape[-1]/scale))))(img)
        #     img = torch.nn.functional.pad(img, (0, self.size[0] - img.shape[-1], 0, self.size[1] - img.shape[-2], 0, 0))

        data[self.target] = img
        return data


class ProduceDescription():

    """
    Method that creates naive text prompts combining a prompt template, atributes (e.g. noisy), and categories
    (e.g. cataract). Also, this method is used to integrate text data with the modality prompt template.
    """

    def __init__(self, caption):
        self.caption = caption

    def __call__(self, data):

        # Create text
        atr_sample = random.sample(data['atributes'], 1)[0] if len(data['atributes']) > 0 else ""
        cat_sample = random.sample(data['categories'], 1)[0] if len(data['categories']) > 0 else ""

        data["sel_category"] = cat_sample
        if 'OCT' in atr_sample:
            data["report"] = ['An Optical Coherence Tomography Image shows '+cat_sample.lower()]
        else:
            data["report"] = [self.caption.replace("[ATR]",  atr_sample).replace("[CLS]",  cat_sample).replace("  ", " ")]

        return data


class AugmentDescription():

    """
    Method that augments naive text prompts into expert knowledge prompts by changing the category name
    by expert descriptions of the target category.
    """

    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, data):

        if self.augment:
            if data["image_name"].split("/")[0] not in ["00_OCTCELL", "06_EYENET", "11_STARE", "08_ODIR-5K", "31_JICHI"]:
                if data["sel_category"] in list(definitions.keys()):
                    prompts = [data["sel_category"]] + definitions[data["sel_category"]]
                    new_cat = random.sample(prompts, 1)[0]
                    data["report"][0] = data["report"][0].replace(data["sel_category"], new_cat)
                    data["augmented_category"] = new_cat

        return data


class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d


class SelectRelevantKeys():

    def __init__(self, target_keys=None):
        if target_keys is None:
            target_keys = ['image', 'report', 'sel_category', 'atributes']
        self.target_keys = target_keys

    def __call__(self, data):
        d = {key: data[key] for key in self.target_keys}
        return d