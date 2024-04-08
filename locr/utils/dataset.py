"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import logging
import os
from math import prod
from pathlib import Path
from functools import partial
import random
from typing import Dict, Tuple, Callable
from PIL import Image, UnidentifiedImageError

import torch
import fitz
import json
import orjson
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from locr.dataset.rasterize import rasterize_paper


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for processing a list of images using a preparation function.

    This dataset takes a list of image paths and applies a preparation function to each image.

    Args:
        img_list (list): List of image paths.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        img_list (list): List of image paths.
        prepare (Callable): The preparation function.
    """

    def __init__(self, img_list, prepare: Callable):
        super().__init__()
        self.img_list = img_list
        self.prepare = prepare

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            return self.prepare(img)
        except:
            return


class LazyDataset(Dataset):
    """
    Lazy loading dataset for processing PDF documents.

    This dataset allows lazy loading of PDF documents and provides access to processed images
    using a specified preparation function.

    Args:
        pdf (str): Path to the PDF document.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        name (str): Name of the PDF document.
    """

    def __init__(self, pdf, prepare: Callable):
        super().__init__()
        self.prepare = prepare
        self.name = str(pdf)
        self.init_fn = partial(rasterize_paper, pdf)
        self.dataset = None
        self.size = len(fitz.open(pdf))

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i == 0 or self.dataset is None:
            self.dataset = ImageDataset(self.init_fn(), self.prepare)
        if i <= self.size and i >= 0:
            return self.dataset[i], self.name if i == self.size - 1 else ""
        else:
            raise IndexError

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            _batch = []
            for i, x in enumerate(batch):
                image, name = x
                if image is not None:
                    _batch.append(x)
                elif name:
                    if i > 0:
                        _batch[-1] = (_batch[-1][0], name)
                    elif len(batch) > 1:
                        _batch.append((batch[1][0] * 0, name))
            if len(_batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(_batch)
        except AttributeError:
            pass


class SciPDFDataset(Dataset):
    """
    Custom dataset for scientific PDF data.

    This dataset loads data from JSONL files and provides access to images, ground truth,
    and metadata.

    Args:
        path_to_index (str): Path to the index file.
        split (str, optional): Split of the dataset (e.g., "train", "test"). Default is "train".
        root_name (str, optional): Root directory name. Default is an empty string.
        template (str, optional): Template for split naming. Default is "%s".

    Attributes:
        empty_sample: Placeholder for empty samples.
    """

    empty_sample = None

    def __init__(
        self,
        path_to_index: str,
        split: str = "train",
        root_name="",
        template="%s",
    ) -> None:
        super().__init__()
        self.path_to_index = Path(path_to_index)    # path_to_index: .jsonl的path
        self.root_name = root_name
        self.path_to_root = self.path_to_index.parent
        if not split in self.path_to_index.stem:    # split=validation, file=validation.jsonl
            pti = self.path_to_index.with_stem(self.path_to_index.stem.replace('train','validation'))
            if pti.exists():
                self.path_to_index = pti
            else:
                raise ValueError(f'Dataset file for split "{split}" not found: {pti}')
        self.dataset_file = None  # mulitprocessing
        # load seek map
        seek_path = self.path_to_root / (self.path_to_index.stem + ".seek.map") # 和jsonl同名、同目录的.seek.map文件
        if seek_path.exists():
            self.seek_map = orjson.loads(seek_path.open().read())
        else:
            raise ValueError(
                'No "%s" found in %s' % (seek_path.name, str(self.path_to_root))
            )
        self.dataset_length = len(self.seek_map)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict:
        position = self.seek_map[index]
        with open(self.path_to_index,'r') as fi:
            self.dataset_file = fi
            self.dataset_file.seek(position)
            line = self.dataset_file.readline()
        try:
            data: Dict = json.loads(line)
        except Exception as e:
            logging.info(
                "JSONL for sample %i could not be loaded at position %i: %s\n%s",
                index,
                position,
                str(e),
                line[:100],
            )
            return self.empty_sample
        img_path: Path = self.path_to_root / self.root_name / data.pop("image")
        if not img_path.exists():
            logging.info("Sample %s could not be found.", img_path)
            return self.empty_sample
        try:
            img = Image.open(img_path)
        except UnidentifiedImageError:
            logging.info("Image %s could not be opened.", img_path)
            return self.empty_sample
        return {"image": img, "prompt":data.pop("prompt"),"label": data.pop("label"),"pretext":data.pop("pretext"), "meta": data}

    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]


class LOCRDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(
        self,
        dataset_path: str,
        locr_model: PreTrainedModel,
        max_length: int,
        prompt_label_length: int = 1,
        split: str = "train",
        root_name: str = "",
    ):
        super().__init__()
        self.locr_model = locr_model
        self.max_length = max_length
        self.prompt_label_length = prompt_label_length
        self.split = split
        self.perturb = "PERTURB" in os.environ and os.environ["PERTURB"]
        # TODO improve naming conventions
        template = "%s"
        self.dataset = SciPDFDataset(
            dataset_path, split=self.split, template=template, root_name=root_name
        )
        self.dataset_length = len(self.dataset)
        self.pad_id = self.locr_model.decoder.tokenizer.pad_token_id
        self.eos_id = self.locr_model.decoder.tokenizer.eos_token_id
        self.global_start_id = self.locr_model.decoder.model.config.decoder_start_token_id  
        self.gloabl_eos_id = 10   
        if self.perturb:
            print("Perturb")

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
        """
      
        sample = self.dataset[idx]  
        if sample is None:
            # if sample is broken choose another randomly
            return self[random.randint(0, self.dataset_length - 1)]
        if sample is None or sample["image"] is None or prod(sample["image"].size) == 0:
            input_tensor = None
        else:
            input_tensor = self.locr_model.encoder.prepare_input(
                sample["image"], random_padding=self.split == "train"
            )
        tokenizer_out = self.locr_model.decoder.tokenizer(
                sample["pretext"]+sample['label'], 
                return_token_type_ids=False,
            )
       
        pre_ids = []
        prompts = []
        pre_id_lst = tokenizer_out["input_ids"]   
        pre_ids.append(pre_id_lst[0][0])
        pre_ids.append(pre_id_lst[0][0])  
        prompts.append([[0,0],[0,0]])   

        skipto = -1
        for i,pre_id in enumerate(pre_id_lst):  
            if self.split == "train" and self.perturb:
                if i <= skipto:
                    continue
                else:
                    if i > 5 and random.random() < 0.02:
                        skiplen = random.choices([0,1], weights=[5,1], k=1)[0]
                        if i + skiplen < len(pre_id_lst) - 5:
                            skipto = i + skiplen
                            continue

            if pre_id[1] == 243:    # 去除单独空格
                pre_id = pre_id[2:-1]
            else:
                pre_id = pre_id[1:-1]   # 去掉<s>和</s>
            pre_ids.extend(pre_id)
            prompt = [sample['prompt'][i]]*len(pre_id)     # 一个word可能被拆分成多个token
            prompts.extend(prompt)
        # truncation
        pre_ids = pre_ids[:self.max_length-1]
        pre_ids.append(pre_id_lst[-1][-1]) 
        prompts = prompts[:self.max_length-2]
        prompts.extend([[[0.99,0.99],[1,1]]]*2)   
        attention_mask = [1]*len(pre_ids)
        # padding
        prompts = prompts + [[[0,0],[0,0]]]*max(0,self.max_length-len(pre_ids))
        attention_mask = attention_mask + [0]*max(0,self.max_length-len(pre_ids))
        pre_ids = pre_ids + [self.pad_id]*max(0,self.max_length-len(pre_ids))
        # to_tensor
        prompts = torch.tensor(prompts)        
        attention_mask = torch.tensor(attention_mask[:-1])  
        label_ids = torch.tensor(pre_ids[1:])   
        pre_ids = torch.tensor(pre_ids[:-1])
        
        assert pre_ids.shape[0]==attention_mask.shape[0]==prompts.shape[0]-1
        
       
                
        # randomly perturb ground truth tokens
        if self.split == "train" and self.perturb:
            # check if we perturb tokens
            unpadded_length = attention_mask.sum()
            while random.random() < 0.1:
                try:
                    pos = random.randint(1, unpadded_length - 2)
                    if random.random() < 0.5:   # 50%概率替换token
                        token = random.randint(
                            23, len(self.locr_model.decoder.tokenizer) - 1
                        )
                        pre_ids[pos] = token
                    else:   # 50%概率替换box
                        pos2 = random.randint(1, unpadded_length - 2)
                        prompts[pos] = prompts[pos2]
                except ValueError:
                    break
        
       
        return input_tensor, pre_ids, attention_mask, label_ids, prompts
