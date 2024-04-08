# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from locr.visualization import visual_box
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d

# 这个 PromptEncoder 类实现了 prompt 的编码,为 mask解码器 提供prompt输入。它包含:
# __init__ 方法:
# 1. 输入参数:
#     - embed_dim: prompt 的 embedding 维度
#     - image_embedding_size: 图像 embedding 的空间大小，表示为(H, W)
#     - input_image_size: 输入到图像编码器的填充后的图像尺寸，表示为 (H, W)。
#     - mask_in_chans: 用于编码输入掩码的隐藏通道数
#     - activation: 用于编码输入掩码的激活函数
# 2. 记录 embed_dim、image_size 和 image_embedding_size。 
# 3. 实例化 PositionEmbeddingRandom 作为位置 embedding 层 pe_layer。
# 4. 实例化 4个 Embedding 层作为点 prompt 的 embedding,以及 not_a_point_embed 用于非点 prompt。
# 5. 计算掩码输入大小 mask_input_size 为 (4 * image_embedding_size[0], 4 * image_embedding_size[1])。
# 6. 实例化 mask_downscaling 为多个 Conv2d 和 LayerNorm2d 层,用于下采样和编码输入掩码。
# 7. 实例化 no_mask_embed 用于无掩码 prompt 的 embedding。
class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size            # 原始image_size ； 若坐标已归一化，输入(1,1)
        self.image_embedding_size = image_embedding_size    # encoder输出的image_size
        self.scaling_size = input_image_size[0]//image_embedding_size[0]
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  #  2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
      
    
    # 这个 get_dense_pe 方法的作用是返回用于对点 prompt 进行编码的密集位置编码。它包含:
    # 1. 调用 pe_layer(image_embedding_size) 得到形状为 (embed_dim)x(embedding_h)x(embedding_w) 的位置编码,
    # image_embedding_size 是图像 embedding 的空间大小。
    # 2. 使用 unsqueeze(0) 增加 batch 维度,得到形状为 1x(embed_dim)x(embedding_h)x(embedding_w) 的位置编码。
    # 3. 返回该位置编码用于对点 prompt 进行编码。
    # 所以,这个 get_dense_pe 方法的作用就是返回一个密集的位置编码,该位置编码具有和图像 embedding 相同的空间尺寸,
    # 用于对点 prompt 进行位置编码,从而得到丰富的 prompt 表达。
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    # 这个 _embed_points 方法的作用是对点 prompt 进行 embedding。它包含:
    # 1. 将 points 中的坐标增加 0.5,将其移至像素中心。
    # 2. 如果 pad 为 True,则会在 points 上追加一个坐标为 [0,0] 和 label 为 -1 的额外点, 并相应地扩充labels。这是用于当未提供 bbox 时的补齐。
    # 3. 调用 pe_layer.forward_with_coords 对 points 进行位置编码,得到 point_embedding。
    # 4. 将 point_embedding 中 label 为 -1 的点 embedding 设置为0。
    # 5. 将 point_embedding 中label为 -1 的点 embedding 增加 not_a_point_embed 的权重。
    # 6. 根据 label 为 0 或 1, 将相应的 point_embedding 增加 point_embeddings[0] 或 point_embeddings[1] 的权重。
    # 7. 返回 point_embedding 作为点 prompt 的 embedding。
    # 所以, 这个 _embed_points 方法实现了对点 prompt 的完整 embedding 过程。
    # 它包含位置编码、分类 Embedding 和类别偏置, 可以得到表达丰富的点 prompt embedding。
    def _embed_points(
        self,
        points: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        bs,seq_len = points.shape[0],points.shape[1]    # [bs,len,1,2]
        points = points + 0.5  # Shift to center of pixel?
        points = points.reshape(-1, 1, 2)    # [bs*len,1,2]
        # 其中当 prompt 不提供 bbox 时，会在提供 point 上再追加点 [0,0]，label=-1 的哑 point，
        # 使用的类别特征 embedding 为上边的代码的 self.not_a_point_embed
        if pad: # padding_point:torch.tensor [bs,1,2]
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device,dtype=torch.float32)
            # points: torch.tensor: [bs,n,2]:
            points = torch.cat([points, padding_point], dim=1)  #[bs*len,1+1,2]
        # 位置编码
        point_embedding = self.pe_layer.forward_with_coords(points)
        point_embedding = point_embedding.reshape(bs,seq_len,2,-1)
        return point_embedding

    # 这个 _embed_boxes 方法的作用是对框 prompt 进行 embedding。它包含:
    # 1. 将 boxes 中的坐标增加 0.5, 将其移至像素中心。
    # 2. 将 boxes reshape 为形状为 (-1, 2, 2) 的张量 coords, 包含框的左上角和右下角坐标。
    # 3. 调用 pe_layer.forward_with_coords 对 coords 进行位置编码,得到 corner_embedding。
    # 4. 将 corner_embedding 中的第 0 维(左上角)增加 point_embeddings[0] 的权重。
    # 5. 将 corner_embedding 中的第 1 维(右下角)增加 point_embeddings[1] 的权重。 
    # 6. 返回 corner_embedding 作为框 prompt 的 embedding。
    # 所以, 这个 _embed_boxes 方法实现了对框 prompt 的 embedding。它对框的左上角和右下角坐标进行了位置编码,
    # 并增加相应的角点 Embedding, 可以得到表达丰富的框 prompt embedding。
    # 这个 _embed_boxes 方法提供了框 prompt embedding 的详细实现, 
    # 包含位置编码和框角点 Embedding, 是理解框 prompt 表达的基础。
    # 总的来说,这个 _embed_boxes 方法实现了框 prompt 的 EMBEDDING 过程,
    # 可以获取表达丰富的框 prompt embedding, 为掩码解码器提供有效的 prompt 输入。
    # 这个 _embed_boxes 方法与 _embed_points 方法一起,
    # 实现了对点 prompt 和框 prompt 的完整 embedding 流程,
    # 可以为掩码解码器提供丰富多样的 prompt 表达, generate高质量的掩码输出。
    def _embed_boxes(self, boxes: torch.Tensor,omit_ratio=0) -> torch.Tensor:
        """Embeds box prompts."""
        bs,seq_len = boxes.shape[0],boxes.shape[1]
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)    # [bs*len,2,2]
        corner_embedding = self.pe_layer.forward_with_coords(coords,omit_ratio=omit_ratio)    # corner_embedding:[bs*seq_len,2,1024]；coords<0(==-1)的emb为0,omit_ratio比例的为0
        mask = (corner_embedding.reshape(bs*seq_len,-1)[:,:] == 0).all(dim=1)   # mask:[bs*seq_len]
        corner_embedding[~mask, 0, :] += self.point_embeddings[0].weight    # left-up
        corner_embedding[~mask, 1, :] += self.point_embeddings[1].weight    # right-bottom
        corner_embedding = corner_embedding.reshape(bs,seq_len,2,-1) # [bs*seq_len,2,d] -> [bs,seq_len,2,d]
        return corner_embedding

  

    # 这个 _get_batch_size 方法的作用是根据 prompt 输入计算输出的 batch size。它包含:
    # 1. 如果 points 不为 None,则返回 points[0] 的第 0 维作为 batch size。points[0] 中包含 prompt 坐标。
    # 2. 如果 boxes 不为 None,则返回 boxes 的第 0 维作为 batch size。boxes 中包含 prompt 框选坐标。
    # 3. 如果 masks 不为 None,则返回 masks 的第 0 维作为 batch size。masks 中包含 prompt 掩码输入。
    # 4. 否则返回 1 作为 batch size。
    # 所以,这个 _get_batch_size 方法根据是否输入了点 prompt、框 prompt或掩码 prompt,返回相应的batch size。
    # 如果未输入任何 prompt,则返回 1 作为 batch size。
    # 这个 _get_batch_size 方法提供了根据 prompt 输入推断输出 batch size的 简单实现,是设计基于 prompt 的生成模型的常用技巧。
    def _get_batch_size(
        self,
        points: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:  # [bs.len,1,2]
            return points.shape[0],points.shape[1]
        elif boxes is not None: # [bs,len,2,2]
            return boxes.shape[0],boxes.shape[1]
        else:
            return 1,1

    # 这个 _get_device 方法的作用很简单,就是返回点 prompt 的第一个 Embedding 层 point_embeddings[0]
    # 的权重参数 weight 所在的设备,作为 PromptEncoder 的设备。
    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def add_random_noise(
        self,
        boxes):
        '''
        add random noise to box,
        normal random with standard_variance=0.8*box sidelength
        '''
        bs,seq_len = boxes.shape[0],boxes.shape[1]
        boxes = boxes.reshape(-1,2,2)   # [bs,len,2,2] -> [bs*seq_len,2,2]
        valid_mask = torch.unique(torch.where(torch.diff(boxes.reshape(-1,4)))[0]) # 除 [0,0,0,0](pad)或[-1,-1,-1,-1](mask)之外的box
        box_width,box_height = boxes[valid_mask,1,0]-boxes[valid_mask,0,0],boxes[valid_mask,1,1]-boxes[valid_mask,0,1]  # width:[bs*seq_len]; height:[bs*seq_len]
        boxes[valid_mask,0,0] += torch.min(torch.normal(0.0,box_width*0.5),box_width*0.5)    # x1
        boxes[valid_mask,0,1] += torch.min(torch.normal(0.0,box_height*0.5),box_height*0.5)   # y1
        boxes[valid_mask,1,0] += torch.max(torch.normal(0.0,box_width*0.5),-box_width*0.5)    # x2
        boxes[valid_mask,1,1] += torch.max(torch.normal(0.0,box_height*0.5),-box_height*0.5)   # y2
        return boxes.reshape(bs,seq_len,2,2)
        
    
    # 这个 forward 方法的作用是对各种 prompt 进行 embedding, 并返回稀疏 embedding 和密集 embedding。它包含:
    # 1. 调用 _get_batch_siz e根据点 prompt、框 prompt和掩码 prompt计算输出的 batch size bs。
    # 2. 初始化稀疏 embedding为形状为 (bs, 0, self.embed_dim) 的空张量, 设备为 _get_device() 的返回设备。
    # 3. 如果 points 不为 None,则调用 _embed_points 对点 prompt 进行 embedding, 
    # 得到 point_embeddings, 并将其拼接到 sparse_embeddings。
    # 4. 如果 boxes 不为 None,则调用 _embed_boxes 对框 prompt 进行 embedding,
    # 得到 box_embeddings, 并将其拼接到 sparse_embeddings。
    # 5. 如果 masks 不为 None, 则调用 _embed_masks 对掩码 prompt 进行 embedding, 得到 dense_embeddings。
    # 6. 否则, 将 no_mask_embed 的权重 reshape 并扩展为形状为 (bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
    # 的张量作为 dense_embeddings。
    # 7. 返回 sparse_embeddings 和 dense_embeddings 作为稀疏 embedding 和密集 embedding。
    # 所以,这个 forward 方法实现了对点 prompt、框 prompt 和掩码 prompt 的 embedding, 
    # 可以得到表达丰富的稀疏 embedding 和密集 embedding, 为下游的解码器提供复杂的 prompt 表达。
    # 这个 forward 方法提供了 prompt 的完整 embedding 流程,包含对三种 prompt 的处理, 
    # 可以获得多模态的 prompt 表达,为实现高质量的 prompt 驱动生成模型打下了基础。
    # 总的来说,这个 forward 方法实现了 prompt 的 ENCODING 过程, 可以获取稀疏 embedding 和密集 embedding
    # 两种 prompt 表达,为实现高质量的多模态 prompt 驱动生成模型提供支持。
    def forward(
        self,
        points: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        add_noise: bool = False,
        omit_ratio = 0,
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (torch.Tensor or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
        """
        
        
        bs,seq_len = self._get_batch_size(points, boxes)
        sparse_embeddings = torch.empty((bs, seq_len, 0, self.embed_dim), device=self._get_device(),dtype=torch.float32) #[bs,len,0,d]
        if points is not None:
            coords = points.to(self._get_device())    # [bs,n,2]
            point_embeddings = self._embed_points(coords, pad=(boxes is None))  #[bs,len,1,2]->[bs,len,1+1,d]
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=-2)  #[bs,len,n+1,d]
        if boxes is not None:
            boxes = boxes.to(self._get_device())
            if add_noise:
                boxes = self.add_random_noise(boxes)    #  [bs,len,2,2]
            box_embeddings = self._embed_boxes(boxes,omit_ratio=omit_ratio)   # [bs,len,2,2]->[bs,len,2,d]；coords<0(==-1)的emb为0
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=-2)

        

        return sparse_embeddings    #[bs,len,2,d]


# 这个 PositionEmbeddingRandom 类实现了随机空间频率的位置编码。它包含:



# 所以,这个PositionEmbeddingRandom类实现了对图像坐标点的随机位置编码。它可以对归一化坐标和非归一化坐标进行编码,为PromptEncoder提供位置编码能力,显著丰富prompt的表达。
# 这个PositionEmbeddingRandom类提供了随机位置编码的实现,为PromptEncoder类带来位置表达的能力,可以丰富prompt的表达,提高prompt驱动生成的质量。
# 总的来说,这个PositionEmbeddingRandom类实现了用于prompt位置编码的随机位置编码器,为PromptEncoder类提供位置编码能力,可以丰富prompt的表达,显著提高prompt驱动生成的质量。
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    # __init__ 方法:
    # 1. 输入参数:
    #     - num_pos_feats: 位置编码的特征数
    #     - scale: 位置编码的 scale, 默认为 1.0
    # 2. 注册 buffer positional_encoding_gaussian_matrix 为形状为 (2, num_pos_feats) 的高斯随机矩阵。
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    # _pe_encoding方法:
    # 1. 输入参数 coords 为归一化到 [0,1] 的坐标点。
    # 2. 将 coords 映射到 [-1,1] 区间。
    # 3. 将 coords 与 positional_encoding_gaussian_matrix 相乘。
    # 4. 将结果乘以 2*π。
    # 5. 拼接 sin 和 cos 作为位置编码,返回形状为 (d_1, ..., d_n, C) 的张量。
    def _pe_encoding(self, coords: torch.Tensor, omit_ratio=0) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords_emb = 2 * coords - 1 # coords: [bs*seq_len,2,2]
        coords_emb = coords_emb @ self.positional_encoding_gaussian_matrix  # positional_encoding_gaussian_matrix:[2,d/2]
        coords_emb = 2 * np.pi * coords_emb #[bs,n+1,d/2]
        coords_emb = torch.cat([torch.sin(coords_emb), torch.cos(coords_emb)], dim=-1)
        # coords<0 => mask_prompt=-1 => position_emb=0
        mask = (coords.reshape(-1,4)[:,:] < 0).all(dim=1)      # mask : [bs*len],coord < 0(==-1)的emb为0
        coords_emb[mask, :, :] = 0      
        # 随机选取一部分有prompt输入的数据，隐藏其prompt输入，但保留prompt_true计算loss
        random_numbers = torch.rand(coords.shape[0],device=coords.device)
        omit_mask = random_numbers < omit_ratio
        coords_emb[omit_mask,:,:] = 0
        return coords_emb    # [bs*len,n+1,d]

    # forward 方法:
    # 1. 输入参数 size 为 (H, W) 的网格大小。
    # 2. 生成一个形状为 (H, W) 的网格,并获得 y 和 x 轴的顺序编码。
    # 3. 归一化 y_embed 和 x_embed 到 [0,1] 区间。
    # 4. 调用 _pe_encoding 对 x_embed 和 y_embed 进行位置编码。
    # 5. 返回位置编码,形状为 (C, H, W) 。
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    # forward_with_coords方法:
    # 1. 输入参数 coords_input 为未归一化到 [0,1] 的坐标, image_size 为 (H, W) 的图像大小。
    # 2. 归一化 coords_input 到 [0,1] 区间。
    # 3. 调用 _pe_encoding 对 coords 进行位置编码。
    # 4. 返回位置编码,形状为 (B, N, C)。
    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int] = None, omit_ratio = 0
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()           # [bs*seq_len,2,2]
        # image_size is None: 传入坐标已归一化
        if image_size is not None:
            coords[:, :, 0] = coords[:, :, 0] / image_size[1]
            coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords,omit_ratio=omit_ratio)  # coords < 0(==-1)的emb为0