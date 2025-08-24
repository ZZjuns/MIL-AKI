"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import math
from abc import ABC
from typing import Dict, Optional
import torch
from torch import nn

# from millet.model.backbone.Xgboost import num_classes
from millet.model.backbone.nystrom_attention import NystromAttention


class MILPooling(nn.Module, ABC):
    """Base class for MIL pooling methods."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_clz = n_clz
        self.dropout_p = dropout
        self.apply_positional_encoding = True
        # Create positional encoding and dropout layers if needed
        if apply_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)


class GlobalAveragePooling(MILPooling):
    """GAP (EmbeddingSpace MIL) pooling."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0,
        apply_positional_encoding: bool = False,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.bag_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits and interpretation (CAM).
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Swap instance embeddings back after adding positional embeddings (if using). Needed for CAM
        instance_embeddings = instance_embeddings.transpose(2, 1)
        # Calculate class activation map (CAM)
        cam = self.bag_classifier.weight @ instance_embeddings
        # Mean pool (GAP) to bag embeddings
        bag_embeddings = instance_embeddings.mean(dim=-1)
        # Classify the bag embeddings
        bag_logits = self.bag_classifier(bag_embeddings)
        return {
            "bag_logits": bag_logits,
            "interpretation": cam,
        }


class MILInstancePooling(MILPooling):
    """Instance MIL pooling. Instance prediction then averaging."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits and interpretation (instance predictions).
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Classify instances
        instance_logits = self.instance_classifier(instance_embeddings)
        # Mean pool to bag prediction
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": instance_logits.transpose(1, 2),
        }


class MILAttentionPooling(MILPooling):
    """Attention MIL pooling. Instance attention then weighted averaging of embeddings."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.bag_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits and interpretation (attention).
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Use attention to get bag embedding
        instance_embeddings = instance_embeddings * attn
        bag_embedding = torch.mean(instance_embeddings, dim=1)
        # Classify the bag embedding
        bag_logits = self.bag_classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            # Attention is not class wise, so repeat for each class
            "interpretation": attn.repeat(1, 1, self.n_clz).transpose(1, 2),
        }


class MILAdditivePooling(MILPooling):
    """Additive MIL pooling. Instance attention then weighting of embeddings before instance prediction."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits, interpretation (instance predictions weight by attention),
        unweighted instance logits, and attn values.
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Scale instances and classify
        instance_embeddings = instance_embeddings * attn
        instance_logits = self.instance_classifier(instance_embeddings)
        # Mean pool to bag prediction
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": (instance_logits * attn).transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2,dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class MILConjunctivePooling(MILPooling):
    """Conjunctive MIL pooling. Instance attention then weighting of instance predictions."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 24,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )

        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 2),
            nn.Sigmoid(),
        )
        # todo 改
        self.instance_classifier = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(d_in, n_clz)
            nn.Linear(d_in, 2)
        )
        # self.instance_classifier = nn.Linear(d_in, n_clz)
        #todo 改
        initialize_weights(self)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits, interpretation (instance predictions weight by attention),
        unweighted instance logits, and attn values.
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        # instance_embeddings = instance_embeddings.transpose(1, 2)

        # Add positional encodings
        # if self.apply_positional_encoding:
            #todo 改
            # instance_embeddings = self.positional_encoding(instance_embeddings, pos)
            # instance_embeddings = self.pos_layer(instance_embeddings, self.wave1,self.wave2,self.wave3)

        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Classify instances
        instance_logits = self.instance_classifier(instance_embeddings)

        # Weight and sum
        weighted_instance_logits = instance_logits * attn
        bag_logits = torch.mean(weighted_instance_logits, dim=1)

        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_instance_logits.transpose(1, 2),
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class PositionalEncoding(nn.Module):
    """
    Adapted from (under BSD 3-Clause License):
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Batch, ts len, d_model
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor, x_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply positional encoding to a set of time series embeddings.

        :param x: Embeddings.
        :param x_pos: Optional positions (indices) of each timestep. If not provided, will use range(len(time series)),
        i.e. 0,...,t-1.
        :return: A tensor the same shape as x, but with positional encoding added to it.
        """
        if x_pos is None:
            x_pe = self.pe[:, : x.size(1)]
        else:
            x_pe = self.pe[0, x_pos]
        x = x + x_pe
        return x


### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift):  # size :d*kernelsize  scale:d*1 shift:d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (int): Size of the kernel.
    scale (float): Scale of the wavelet.
    shift (float): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """

    x = torch.linspace(-(size[1] - 1) // 2, (size[1] - 1) // 2, size[1]).cuda()
    # print(x.shape)
    x = x.reshape(1, -1).repeat(size[0], 1)
    # print(x.shape)
    # print(shift.shape)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / (3 ** 0.5 * torch.pi ** 0.25)
    wavelet = C * (1 - (x / scale) ** 2) * torch.exp(-(x / scale) ** 2 / 2) * 1 / (torch.abs(scale) ** 0.5)

    return wavelet  # d*L

class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len=256, hidden_len=512, dropout=0.0):
        super().__init__()

        # n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)

    def forward(self, x, wave1, wave2, wave3):
        x = x.transpose(1, 2)

        D = x.shape[1]
        scale1, shift1 = wave1[0, :], wave1[1, :]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D, 19), scale=scale1, shift=shift1)
        scale2, shift2 = wave2[0, :], wave2[1, :]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D, 19), scale=scale2, shift=shift2)
        scale3, shift3 = wave3[0, :], wave3[1, :]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D, 19), scale=scale3, shift=shift3)

        # Eq. 11
        pos1 = torch.nn.functional.conv1d(x, wavelet_kernel1.unsqueeze(1), groups=D, padding='same')
        pos2 = torch.nn.functional.conv1d(x, wavelet_kernel2.unsqueeze(1), groups=D, padding='same')
        pos3 = torch.nn.functional.conv1d(x, wavelet_kernel3.unsqueeze(1), groups=D, padding='same')
        x = x.transpose(1, 2)  # B*N*D
        # print(x.shape)

        # Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2) + pos2.transpose(1, 2) + pos3.transpose(1, 2))  # + mixup_encording

        # mixup token information
        return x

