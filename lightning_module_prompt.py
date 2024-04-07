"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import math
import random
from pathlib import Path
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from nougat import PromptNougatConfig, PromptNougatModel
from nougat.metrics import get_metrics
from nougat.cal_loss import cal_loss


class PromptModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.validation_step_outputs = []
        self.config = config
        if self.config.get("model_path", False):
            self.model = PromptNougatModel.from_pretrained(
                self.config.model_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                window_size=self.config.window_size,
                encoder_layer=self.config.encoder_layer,
                decoder_layer=self.config.decoder_layer,
                patch_size=self.config.patch_size,
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                hidden_dimension=self.config.hidden_dimension,
                ignore_mismatched_sizes=True,
            )
            if self.config.resume_from_checkpoint_path is not None:
                print('not none')
            else:
                print('none')
    
        else:
            self.model = PromptNougatModel(
                config=PromptNougatConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    window_size=self.config.window_size,
                    encoder_layer=self.config.encoder_layer,
                    decoder_layer=self.config.decoder_layer,
                    tokenizer_file=self.config.tokenizer,
                    patch_size=self.config.patch_size,
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    hidden_dimension=self.config.hidden_dimension,
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors, pre_input_ids, label_ids, attention_masks,prompts = list(), list(), list(), list(), list()
        if batch is None:
            return
        for batch_data in batch:    # batch: input_tensor, pre_ids, attention_mask, label_id, prompt
            if batch_data is None or batch_data[0] is None:
                continue
            image_tensors.append(batch_data[0])     # image
            pre_input_ids.append(batch_data[1])     # pre_ids
            attention_masks.append(batch_data[2])   # attention_mask
            label_ids.append(batch_data[3])         # label_id
            prompts.append(batch_data[4])           # prompt
            
        image_tensors = torch.cat(image_tensors)
        pre_input_ids = torch.cat(pre_input_ids)
        attention_masks = torch.cat(attention_masks)
        label_ids = torch.cat(label_ids)
        prompts = torch.cat(prompts)
        
        prompt_in=prompts[:,:-1,:,:].clone().detach()
        prompt_true=prompts[:,1:,:,:].clone().detach()
        
        current_step = self.global_step
        loss_txt,loss_math,loss_table,loss_start,diou,iou,fl = self.model(
                                                                    image_tensors, 
                                                                    pre_input_ids, 
                                                                    attention_masks,
                                                                    label_ids, 
                                                                    prompt_in=prompt_in,
                                                                    prompt_true=prompt_true,
                                                                    current_step=current_step,
                                                                    full_prompt_in = prompt_in)[0] #CrossEntropyLoss()+IoU_loss()

        self.log_dict({"train/loss_txt": loss_txt}, sync_dist=False)
        # loss_token
        weight_start,weight_math,weight_table = 1,1,1
        loss_token = loss_txt
        weight_sum = 1
        if loss_math is not None:   
            self.log_dict({"train/loss_math": loss_math}, sync_dist=False)
            loss_token += weight_math*loss_math
            weight_sum += weight_math
        if loss_table is not None:
            self.log_dict({"train/loss_table": loss_table}, sync_dist=False)
            loss_token += weight_table*loss_table
            weight_sum += weight_table
        if loss_start is not None:
            self.log_dict({"train/loss_start": loss_start}, sync_dist=False)
            loss_token += weight_start*loss_start
            weight_sum += weight_start
        loss_token /= weight_sum
        self.log_dict({"train/loss_token": loss_token}, sync_dist=False)
       
        weight_token,weight_position = 5, 1
        weight_diou,weight_cls = 0.3,1
        if iou is not None:
            loss_position = weight_diou*diou + weight_cls*fl
            loss = (weight_token*loss_token + weight_position*loss_position)/(weight_token+weight_position)
            self.log_dict({"train/loss": loss}, sync_dist=False)            
            self.log_dict({"train/loss_position": loss_position}, sync_dist=False)
            self.log_dict({"train/iou": iou}, sync_dist=False)
            self.log_dict({"train/focal_loss": fl}, sync_dist=False)
            return loss
        else:
            return loss_token

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if batch is None:
            return
        image_tensors,input_ids, attention_masks, labels, prompts = batch
        prompt_in = prompts[:,:-1,:,:].clone().detach()
        prompt_true=prompts[:,1:,:,:].clone().detach()
        if image_tensors is None:
            return
        current_step = self.global_step
       
        output = self.model.inference(
            image_tensors=image_tensors,    # shape=[bs,3,588,1024]
            input_ids = input_ids,
            attention_mask=attention_masks,
            return_attentions=True,
            prompt=prompt_in,
            validation=True,
            current_step = current_step
        )
       
        
       
        gts = self.model.decoder.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        ) 
        preds = output["predictions"]
                                        
        metrics = get_metrics(gts, preds, pool=False)
        scores = {
            "val/" + key: sum(values) / len(values) for key, values in metrics.items()
        }
            
        self.validation_step_outputs.append(scores)
      
        return scores
        

    def on_validation_epoch_end(self):
        if (
            self.validation_step_outputs is not None
            and len(self.validation_step_outputs) >= 1
        ):
            self.log_dict(self.validation_step_outputs[0], sync_dist=True)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert (
                len(self.config.train_batch_sizes) == 1
            ), "Set max_epochs only if the number of datasets is 1"
            steps = self.config.num_training_samples_per_epoch
            max_iter = (self.config.max_epochs * steps) / max(
                1,
                (
                    self.config.train_batch_sizes[0]
                    * torch.cuda.device_count()
                    * self.config.get("num_nodes", 1)
                ),
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = (
                min(self.config.max_steps, max_iter) \
                if max_iter is not None \
                else self.config.max_steps
            )

        assert max_iter is not None
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,self.parameters()), lr=self.config.lr,weight_decay=self.config.weight_decay)
        scheduler = {
            "scheduler": self.exponential_scheduler(
                optimizer,
                self.config.warmup_steps,
                self.config.lr,
                self.config.get("min_lr", 5e-5),
                self.config.get("gamma", 0.9996),
            ),
           
            "name": "learning_rate",
            "interval": "step",
            "frequency": self.config.get("lr_step", 1),
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)   

            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=5e-5, gamma=0.9999):
        def lr_lambda(x):
            if x > warmup_steps or warmup_steps <= 0:
                if lr * gamma ** (x - warmup_steps) > min_lr:
                    return gamma ** (x - warmup_steps)
                else:
                    return min_lr / lr
            else:
                return x / warmup_steps

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = (
            Path(self.config.result_path)
            / self.config.exp_name
            / self.config.exp_version
        )
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class PromptDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
       

    def train_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.train_datasets),
                batch_size=self.train_batch_sizes[0],
                num_workers=self.config.num_workers,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=True,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    def val_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.val_datasets),
                batch_size=self.val_batch_sizes[0],
                pin_memory=True,
                shuffle=False,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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
