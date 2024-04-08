from locr.position_decoder import diou_loss,focal_loss,celoss
from torch.nn import CrossEntropyLoss
import torch
import re


def token_loss(bs,logits,labels,):
    mask_txt = torch.ones_like(labels)
    mask_txt[torch.where(labels==1)] = 0   
    mask_math,mask_table,mask_start = torch.zeros_like(labels),torch.zeros_like(labels),torch.zeros_like(labels)
    # start tokens
    mask_txt,mask_start = mask_txt.reshape(bs,-1),mask_start.reshape(bs,-1)
    mask_txt[:,:5] = 0
    mask_start[:,:5] = 1
    mask_txt,mask_start = mask_txt.view(-1),mask_start.view(-1)
   
    loss_fct = CrossEntropyLoss()
    loss_txt = loss_fct(logits[mask_txt==1],labels[mask_txt==1])    
    loss_math = None
    loss_table = None
    loss_start = loss_fct(logits[mask_start==1],labels[mask_start==1])
    
    return (loss_txt,loss_math ,loss_table, loss_start)

def cal_loss(bs,logits,labels,prompt_pred,prompt_true):
    import time
    begin_time = time.time()
    # loss_token
    loss_txt,loss_math ,loss_table, loss_start = token_loss(bs,logits, labels)   
    
    # loss_position
    valid_mask = torch.unique(torch.where(torch.diff(prompt_true.reshape(-1,4)))[0]) 
    assert  len(prompt_true.shape) == 4, prompt_true.shape
    seq_len = prompt_true.shape[1]
    pre5pos = torch.cat([torch.arange(0, 5) + seq_len * i for i in range(bs)]).to(valid_mask.device)
    pre5pos = torch.where(torch.isin(valid_mask, pre5pos))
    

    if len(valid_mask)>2*bs:   
        box_pred = prompt_pred[0].reshape(-1,2,2)[valid_mask]    # box
        box_true = prompt_true.reshape(-1,2,2)[valid_mask]
        diou,iou = diou_loss(pred=box_pred,target=box_true,pre5pos=pre5pos)  
    
        hm = prompt_pred[1]                                 # [bs,seq_len,28,21]
        hm = hm.reshape(-1,hm.shape[-2],hm.shape[-1])[valid_mask]  # [bs*seq_len,28,21]
        hm_loss = celoss(hm, box_true)
        
        
    else:  
        diou,iou = None,None,None

    
    return loss_txt,loss_math,loss_table,loss_start,diou,iou,hm_loss
