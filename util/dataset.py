from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import tokenizers
import random

from transformers import AutoModel, AutoTokenizer

from util.flair_dataloader import dataloader


# ===========
dataframes_path = 'TODO CHANGE TO YOUR DATAPATH./pretrain_data/fundus_labels/'
data_root_path = 'TODO CHANGE TO YOUR DATAPATH./pretrain_data/fundus_oct_images/'
datasets_debug = ["04_RFMid", "00_OCTCELL"]

datasets_oct = ["00_OCTCELL"]
datasets_fundus = ["01_EYEPACS", "04_RFMid",
                                               "06_DEN", "07_LAG", "08_ODIR", "10_PARAGUAY",
                                               "11_STARE", "12_ARIA", "14_AGAR300", "16_FUND-OCT",
                                               "18_DRIONS-DB", "19_Drishti-GS1",
                                               "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "26_ROC",
                                               "28_OIA-DDR", "30_SUSTech-SYSU", "31_JICHI",
                                               "32_CHAKSU", "33_DR1-2", "35_ScarDat", "36_ACRIMA", "37_DeepDRiD_test", "37_DeepDRiD_train_eval"]
datasets_fundus_oct = ["00_OCTCELL", "01_EYEPACS", "04_RFMid",
                                               "06_DEN", "07_LAG", "08_ODIR", "10_PARAGUAY",
                                               "11_STARE", "12_ARIA", "14_AGAR300", "16_FUND-OCT",
                                               "18_DRIONS-DB", "19_Drishti-GS1",
                                               "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "26_ROC",
                                               "28_OIA-DDR", "30_SUSTech-SYSU", "31_JICHI",
                                               "32_CHAKSU", "33_DR1-2", "35_ScarDat", "36_ACRIMA", "37_DeepDRiD_test", "37_DeepDRiD_train_eval"]

balance = True
batch_size = 16
num_workers = 10
banned_categories = []
caption = "A [ATR] fundus photograph of [CLS]"
augment_description = True
from torchvision.transforms import Compose
from util.flair_dataloader.transforms import LoadImage, ImageScaling, SelectRelevantKeys, CopyDict,\
    ProduceDescription, AugmentDescription
# ===========


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultimodalBertDataset_flair(Dataset):
    def __init__(
        self,
        data_mode,
        max_caption_length: int = 100
    ):
        if data_mode == 'debug':
            datasets_train = datasets_debug
        elif data_mode == 'fundus':
            datasets_train = datasets_fundus
        elif data_mode == 'oct':
            datasets_train = datasets_oct
        elif data_mode == 'fundus_oct':
            datasets_train = datasets_fundus_oct
        
        self.data_list = dataloader.get_data_list(dataframes_path, 
                                   data_root_path, datasets_train, balance, 
                                   batch_size, num_workers, banned_categories, 
                                   caption, augment_description)
        
        self.transforms = Compose([
            CopyDict(),
            LoadImage(),
            ImageScaling(),
            ProduceDescription(caption=caption),
            AugmentDescription(augment=augment_description),
            SelectRelevantKeys()
        ])
        
        self.max_caption_length = max_caption_length
        # self.data_root = data_root
        # self.transform = transform
        # self.images_list, self.report_list = self.read_csv()
        # # random
        # random_seed = 42
        # random.seed(random_seed)
        # random.shuffle(self.images_list)
        # random.seed(random_seed)
        # random.shuffle(self.report_list)
        self.tokenizer = tokenizers.Tokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # self.tokenizer.model_max_length = 77

        # self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        self.tokenizer.enable_padding(length=self.max_caption_length)

    def __len__(self):
        return len(self.data_list)
    
    def _random_mask(self,tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break
            
            if masked_tokens[0][i-1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue
            
            if masked_tokens[0][i-1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index):
        batch = self.transforms(self.data_list[index])
        image = batch['image']
        sent = batch['report'][0]
        data_moda = torch.tensor(0)
        if 'OCT' in batch['atributes']:
            data_moda = torch.tensor(1)

        # image = pil_loader(self.images_list[index])
        # image = self.transform(image)
        # sent = self.report_list[index]
        # sent = '[CLS] '+ sent
        
        encoded = self.tokenizer.encode(sent)
        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)
        masked_ids = self._random_mask(ids)
        return image, ids, attention_mask, type_ids, masked_ids, data_moda
    
    # def read_csv(self):
    #     csv_path = os.path.join(self.data_root,'training.csv')
    #     df = pd.read_csv(csv_path,sep=',')
    #     return df["image_path"], df["report_content"]

    def collate_fn(self, instances: List[Tuple]):
        image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list, datamoda_list = [], [], [], [], [], []
        # flattern
        for b in instances:
            image, ids, attention_mask, type_ids, masked_ids, moda_ids = b
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)
            datamoda_list.append(moda_ids)

        # stack
        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()
        moda_ids_stack = torch.stack(datamoda_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack,
            'tag': moda_ids_stack
        }

        return return_dict