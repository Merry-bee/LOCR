import torch
from torch import nn
import torch.nn.functional as F
import statistics
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import resize, rotate

class PositionDecoder(nn.Module):
    def __init__(self,
                 decoder_attention_heads,
                 decoder_layers,
                 input_dim=588, 
                 hidden_dim=256, 
                 output_dim=5, 
                 num_layers=3,
                 bn_momentum=0.1,
                 image_size=[896,672],
                 scale_factor=2):
        super().__init__()
        import os
        if 'decay' not in os.environ or os.environ['decay'] is None or os.environ['decay'] == '':
            self.decay_rate = 1.0
        else:
            self.decay_rate = float(os.environ['decay'])
        self.image_size = image_size
        self.in_channels = decoder_attention_heads*decoder_layers   # 16*4=64
        self.middle_channels = 16
        self.upscale_factor = scale_factor
        # heatmap: 预测中心点所在网格，最大值为预测点，值为置信度
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),   # in_channels, out_channels, kernel_size, stride=1, padding=(1,0)
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            
            nn.UpsamplingBilinear2d(scale_factor=self.upscale_factor),

            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),   # in_channels, out_channels, kernel_size, stride=1, padding=(1,0)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),   # in_channels, out_channels, kernel_size, stride=1, padding=(1,0)
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.middle_channels, 1, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )
        # 每个点对应bbox的wh
        self.wh_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            
            nn.UpsamplingBilinear2d(scale_factor=self.upscale_factor),

            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.middle_channels, 2, kernel_size=1, stride=1, padding=0),
            )    
        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            
            nn.UpsamplingBilinear2d(scale_factor=self.upscale_factor),
            
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.middle_channels, 2, kernel_size=1, stride=1, padding=0),
            )    
        
    
    def forward(self, heatmap,attention_valid_mask, full_prompt_in = None, image_tensors = None):
        '''
        args:
            heatmap:[bs,16,len(input_ids),588]
            full_prompt_in: 用于惩罚重复生成，在use cache时长度与input_len不同
        '''
        decoder_layers,bs,num_heads,input_len,encoder_len = heatmap.shape  # [4,bs,16,len,588]
        heatmap = heatmap.permute(1,3,4,0,2).reshape(bs,input_len,encoder_len,-1)  # [bs,len,588,4,16] -> [bs,len,588,64]
        
        heatmap = heatmap.reshape(bs*input_len,decoder_layers*num_heads,28,21)   # 输入：(bs,C_in​,H_in,W_in)=[bs*seq_len,4*16,28,21]     
        

        hm = self.cls_head(heatmap)   # 输出：(bs,C_out​,H_out,W_out)=[bs*seq_len,1,28,21] 56,42
        wh = self.wh_head(heatmap)      # [bs*seq_len,2,28,21], 56,42
        offset = self.offset_head(heatmap)  # [bs*seq_len,2,28,21] 56,42

        bs_len, _, output_h, output_w = hm.shape
        

        hm = hm.view([bs_len,-1])
        if full_prompt_in is None:
            # print("Not Decay")
            hmdecay = torch.zeros_like(hm)
        else:
            # print(full_prompt_in.shape)
            cls_in = box2cls(full_prompt_in.reshape(-1, 2,2), output_h, output_w).reshape(bs, -1, output_h,output_w)
            cnt_cls_in = cls_in.cumsum(dim = 1)[:, -input_len:]
            # print("Input Box Count")
            # print(cnt_cls_in)
            cnt_cls_in = torch.where(cnt_cls_in < 5,0,cnt_cls_in)
            hmdecay = cnt_cls_in.reshape(hm.shape).square() * torch.log(torch.tensor(self.decay_rate))
        
        hm += hmdecay.to(hm.device)

        if image_tensors is not None:
            split_size = image_tensors.shape[-1] // output_w
            image_gray = image_tensors.mean(dim=-3)
            image_hsplit = torch.stack(image_gray.split(split_size,dim=-2),dim=1)
            image_split = torch.stack(image_hsplit.split(split_size,dim=-1),dim=2)  # B, output_h, output_w, box_h, box_w
            image_split_std = image_split.std(dim=(-1,-2))
            image_split_std[...,:-1] += image_split_std[...,1:] # shift left
            image_split_std[...,:-1] += image_split_std[...,1:] # shift left
            image_split_std[...,-1,-1] = 1 #0.01/20   # end token
            hm += torch.log(20*image_split_std.clamp(1e-6,0.05)).view([bs_len,-1])

        

        _, indices = torch.max(hm.view([bs_len,-1]),dim=-1)  # [bs*seq_len,1,28,21]->[bs*seq_len]
        indices_x = indices % (output_w)  # [bs*seq_len]
        indices_y = indices // (output_w) # [bs*seq_len] 
        xv = indices_x.float() / (output_w)   # [bs*seq_len]
        yv = indices_y.float() / (output_h)

        indices_bs_len = torch.arange(bs_len,device=heatmap.device)
        
        if self.training:
            xv += offset[indices_bs_len, 0, indices_y, indices_x]
            yv += offset[indices_bs_len, 1, indices_y, indices_x]
        else:
            xv += offset[indices_bs_len, 0, indices_y, indices_x].clamp(0,1.0/output_w)
            yv += offset[indices_bs_len, 1, indices_y, indices_x].clamp(0,1.0/output_h)

        half_w = wh[indices_bs_len, 0, indices_y, indices_x] / 2
        half_h = wh[indices_bs_len, 1, indices_y, indices_x] / 2
        x1 = (xv - half_w).view(bs,input_len)
        y1 = (yv - half_h).view(bs,input_len)
        x2 = (xv + half_w).view(bs,input_len)
        y2 = (yv + half_h).view(bs, input_len)
     
        
        # attention_valid_mask=0 -> pred=[[0,0],[0,0]](pad坐标)
        if attention_valid_mask.shape[1]==4095:   # train/validation with whole sentence
            attention_valid_mask = torch.cat((attention_valid_mask[:,1:],torch.zeros([bs,1]).to(heatmap.device)),dim=1) # attention_mask对应input_ids，对应prompt_true需要向后移动一位
        x1 = torch.mul(x1,attention_valid_mask) # [bs,seq_len]
        x2 = torch.mul(x2,attention_valid_mask)
        y1 = torch.mul(y1,attention_valid_mask)
        y2 = torch.mul(y2,attention_valid_mask)
     
        hm = hm.reshape(bs,input_len,output_h,output_w)  # [bs*seq_len,1,28,21] -> [bs,seq_len,28,21]
        hm = torch.mul(hm,attention_valid_mask[..., None, None])
  
        return [x1,y1,x2,y2],hm
        
def box2cls(box_true,output_h,output_w,gaussian=False,sigma=1.0):
    bs_len = box_true.shape[0]
    cls_true = torch.zeros(bs_len,output_h,output_w, device=box_true.device)   # [bs*seq_len,28,21]
    yv,xv = (box_true[...,0,1]+box_true[...,1,1])/2,(box_true[...,0,0]+box_true[...,1,0])/2,   # 实际坐标(归一化)=[bs*length]
    yv,xv = torch.clamp(yv, 1e-6, 1-1e-6),torch.clamp(xv, 1e-6, 1-1e-6)
    yv,xv = torch.floor(yv*output_h).long(),torch.floor(xv*output_w).long()     # 格子indices
    indices_bs_len = torch.arange(bs_len, device=box_true.device)
    cls_true[indices_bs_len,yv,xv] = 1
    if gaussian:
        # 生成一个高斯分布的核，用于模糊
        x, y = torch.meshgrid(torch.arange(0, output_w), torch.arange(0, output_h))
        kernel = torch.exp(-((x - xv)**2 + (y - yv)**2) / (2 * sigma**2))
        kernel /= kernel.sum()# 归一化核
        cls_true = gaussian_filter(cls_true, sigma=sigma, mode='constant', cval=0)# 将核应用于图像
    
    return cls_true
    

def focal_loss(hm, box_true,gaussian=False):
    bs_len,output_h, output_w = hm.shape  # [bs,seq_len,28,21]
    target = box2cls(box_true,output_h,output_w,gaussian)
    
    pos_inds = target.eq(1).float().to(hm.device)
    neg_inds = target.lt(1).float().to(hm.device)
    hm = nn.Sigmoid()(hm)
    hm = torch.clamp(hm, 1e-6, 1-1e-6)  # clamp to 1e-6 ~ 1-1e-6
    pos_loss = torch.pow(1 - hm, 2) * torch.log(hm) * pos_inds
    neg_loss = torch.pow(hm, 2) * torch.log(1 - hm) * neg_inds 
    
    if gaussian:
        # The negative samples near the positive sample feature point have smaller weights
        neg_weights = torch.pow(1-target, 4)
        neg_loss *= neg_weights
    
    fl =  - 1/torch.numel(hm) * (pos_loss.sum() + neg_loss.sum()) 

    return fl

def celoss(hm, box_true):
    bs_len,output_h, output_w = hm.shape  # [bs,seq_len,28,21]
    pred = hm.view(bs_len,-1)    # [bs_len,588]
    target = box2cls(box_true,output_h,output_w)    # [bs*seq_len,28,21]
    target = target.view(bs_len,-1).max(dim=-1).indices.to(pred.device)  # [bs_len,588]-> [bs_len]
    loss_fct = CrossEntropyLoss()
    celoss = loss_fct(pred,target.view(-1))

    return celoss


def l1_loss(pred, target, mask):
    """
    Calculate l1 loss
    Args:
        pred: offset detection result
        target: offset ground truth
        mask: offset mask, only center point is 1, other place is 0

    Returns: l1 loss

    """
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    # Don't calculate loss in the position without ground truth.
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')

    loss = loss / (mask.sum() + 1e-7)

    return loss
  
def iou(pred,target,epsilon=1e-5):
    '''
    args: 
    pred/target: [bs,length,2,2]
    
    '''
    inter_x1 = torch.max(pred[:,0,0],target[:,0,0])
    inter_y1 = torch.max(pred[:,0,1],target[:,0,1])
    inter_x2 = torch.min(pred[:,1,0],target[:,1,0])
    inter_y2 = torch.min(pred[:,1,1],target[:,1,1])
    # 确保交集面积不小于0
    inter_area = torch.clamp(inter_x2-inter_x1,min=0)*torch.clamp(inter_y2-inter_y1,min=0)
    pred_area = (pred[:,1,0]-pred[:,0,0])*(pred[:,1,1]-pred[:,0,1])
    target_area = (target[:,1,0]-target[:,0,0])*(target[:,1,1]-target[:,0,1])
    union_area = pred_area + target_area - inter_area
    iou = (inter_area/(union_area+epsilon))
    

    return iou
    

def diou_loss(pred,target,pre5pos,prob=None,epsilon=1e-5,alpha=10,y_penalty=2):
    '''
    args: 
    pred/target: [bs,length,2,2]
    
    '''
    pred = pred.reshape(-1,2,2) # [bs*len,2,2]
    target = target.reshape(-1,2,2) 
    
    iou_tensor = iou(pred,target,epsilon) # [bs*len,1]
    
    pred,target = pred,target
    pred_center_x = (pred[:,1,0]+pred[:,0,0])/2
    pred_center_y = (pred[:,1,1]+pred[:,0,1])/2
    target_center_x = (target[:,1,0]+target[:,0,0])/2
    target_center_y = (target[:,1,1]+target[:,0,1])/2
    d2 = (torch.square(pred_center_x-target_center_x)+torch.square(y_penalty*(pred_center_y-target_center_y)))
    out_x1 = torch.min(pred[:,0,0],target[:,0,0])
    out_y1 = torch.min(pred[:,0,1],target[:,0,1])
    out_x2 = torch.max(pred[:,1,0],target[:,1,0])
    out_y2 = torch.max(pred[:,1,1],target[:,1,1])
    c2 = (torch.square(out_x2-out_x1)+torch.square(out_y2-out_y1))
    diou_loss = 1-iou_tensor+alpha*d2
    # prob准确度
    if prob:
        prob_loss = 10*(prob-iou_tensor)**2
        diou_loss += prob_loss
    # 根据位置对每个token加权
    y1=target[torch.where(torch.diff(target[:,0,1],dim=0))[0]][:,0,1]   # 先定位到前后两个token的y值相同的token
    y2=target[torch.where(torch.diff(target[:,0,1],dim=0))[0]][:,1,1]
    mode = statistics.mode([round(h,2) for h in (y2-y1).tolist() if h>0])   # 本页行高：众数
    weights = torch.ones(target.shape[0],dtype=target.dtype).to(diou_loss.device)
    weights[torch.where(torch.diff(target[:,0,1],dim=0)>2*mode)[0]+1] = alpha

    # 前5个token
    weights[pre5pos] = alpha

    diou_loss_mean = (diou_loss * weights).sum() / weights.sum()
    iou_mean = (iou_tensor * weights).sum() / weights.sum()
    
    return diou_loss_mean,iou_mean

         
        

