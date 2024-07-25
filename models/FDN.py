from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn.init as torch_init
import torch.nn as nn
import os
import fairseq
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        # self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        # out = self.mp(out)
        return out


class ASRModel(nn.Module):
    def __init__(self):
        super(ASRModel, self).__init__()
        cp_path = os.path.join('./pretrain_models/xlsr2_300m.pt') 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]

    def forward(self, x):
        emb = self.model(x, mask=False, features_only=True)['x']
        return emb

class GMLPBlock(nn.Module):
    """
    ## gMLP Block

    Each block does the following transformations to input embeddings
    $X \in \mathbb{R}^{n \times d}$ where $n$ is the sequence length
    and $d$ is the dimensionality of the embeddings:

    \begin{align}
    Z &= \sigma(XU) \\
    \tilde{Z} &= s(Z) \\
    Y &= \tilde{Z}V \\
    \end{align}

    where $V$ and $U$ are learnable projection weights.
    $s(\cdot)$ is the Spacial Gating Unit defined below.
    Output dimensionality of $s(\cdot)$ will be half of $Z$.
    $\sigma$ is an activation function such as
    [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).
    """

    def __init__(self, d_model: int, d_ffn: int, seq_len: int,tiny_attn=False):
        """
        `d_model` is the dimensionality ($d$) of $X$
        `d_ffn` is the dimensionality of $Z$
        `seq_len` is the length of the token sequence ($n$)
        """
        super().__init__()
        # Normalization layer fro Pre-Norm
        self.norm = nn.LayerNorm([d_model])
        # Activation function $\sigma$
        self.activation = nn.GELU()
        # Projection layer for $Z = \sigma(XU)$
        self.proj1 = nn.Linear(d_model, d_ffn)
        # Spacial Gating Unit $s(\cdot)$
        self.sgu = SpacialGatingUnit(d_ffn, seq_len,tiny_attn)
        # Projection layer for $Y = \tilde{Z}V$
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
        # Embedding size (required by [Encoder](../models.html#Encoder).
        # We use the encoder module from transformer architecture and plug
        # *gMLP* block as a replacement for the [Transformer Layer](../models.html#Encoder).
        self.size = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `x` is the input embedding tensor $X$ of shape `[seq_len, batch_size, d_model]`
        * `mask` is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other.
        """
        # Keep a copy for shortcut connection
        shortcut = x
        # Normalize $X$
        x = self.norm(x)
        # Projection and activation $Z = \sigma(XU)$
        z = self.activation(self.proj1(x))
        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        z = self.sgu(z, mask)
        # Final projection $Y = \tilde{Z}V$
        z = self.proj2(z)

        # Add the shortcut connection
        return z + shortcut


class SpacialGatingUnit(nn.Module):
    """
    ## Spatial Gating Unit

    $$s(Z) = Z_1 \odot f_{W,b}(Z_2)$$

    where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension,
    and $\odot$ is element-wise multiplication.
    $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).
    """
    def __init__(self, d_z: int, seq_len: int, tiny_attn: bool):
        """
        * `d_z` is the dimensionality of $Z$
        * `seq_len` is the sequence length
        """
        super().__init__()
        # Normalization layer before applying $f_{W,b}(\cdot)$
        # Weight $W$ in $f_{W,b}(\cdot)$.
        self.norm = nn.LayerNorm([d_z // 2])
        # The paper notes that it's important to initialize weights to small values and the bias to $1$,
        # so that during the initial training $s(\cdot)$ is close to identity (apart from the split).
        self.weight = nn.Parameter(torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01), requires_grad=True)
        # Weight $b$ in $f_{W,b}(\cdot)$
        self.tiny_attn=tiny_attn
        # The paper notes that it's important to initialize bias to $1$.
        self.bias = nn.Parameter(torch.ones(seq_len), requires_grad=True)
        self.tn = TinyAttention(d_z, d_z//2)
    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `z` is the input $Z$ of shape `[seq_len, batch_size, d_z]`
        * `mask` is is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other. The last dimension of size `1` is the batch, which we have in other transformer
         implementations and was left for compatibility.
        """

        # Get sequence length
        seq_len = z.shape[0]
        # Split $Z$ into $Z_1$ and $Z_2$
        z1, z2 = torch.chunk(z, 2, dim=-1)

        # Check mask
        if mask is not None:
            # `mask` has shape `[seq_len_q, seq_len_k, batch_size]`.
            # The batch dimension should be of size `1` because this implementation supports
            # only same mask for all samples in the batch.
            assert mask.shape[0] == 1 or mask.shape[0] == seq_len
            assert mask.shape[1] == seq_len
            # Here we only support the same mask for all samples
            assert mask.shape[2] == 1
            # Remove the batch dimension
            mask = mask[:, :, 0]

        # Normalize $Z_2$ before $f_{W,b}(\cdot)$
        z2 = self.norm(z2)
        # Get the weight matrix; truncate if larger than `seq_len`
        weight = self.weight[:seq_len, :seq_len]
        # Apply mask to the weights.
        #
        # If $W_{i,j}$ is $0$ then $f_{W,b}(Z_2)_i$ will not get any information
        # from token $j$.
        if mask is not None:
            weight = weight * mask
        z2 = torch.einsum('ij,jbd->ibd', weight, z2) + self.bias[:seq_len, None, None]
        if self.tiny_attn:
            tn=self.tn(z)
            z2=z2+tn
        # $f_{W,b}(Z_2) = W Z_2 + b$

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1 * z2

class aMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, seq_len: int, gmlp_layers = 1, batch_first=True,
            flag_pool = 'None'):
        """
        gmlp_layer: number of gmlp layers
        d_model: dim(d) of input [n * d]
        d_ffn: dim of hidden feature
        seq_len: the max of input n. for mask 
        batch_first: 
        """
        super().__init__()
        self.batch_first = batch_first
        self.flag_pool = flag_pool
        if(d_ffn > 0):
            pass
        elif(d_ffn < 0):
            #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
            d_ffn = int(d_model / abs(d_ffn))
        layers = []
        for i in range(gmlp_layers):
            layers.append(GMLPBlock(d_model, d_ffn, seq_len,tiny_attn=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if(self.batch_first):
            x = x.permute(1, 0, 2)
            x = self.layers(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.layers(x)
        return x    


class TinyAttention(torch.nn.Module):
    def __init__(self, d_in, d_out=None, d_attn=64):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out or d_in
        self.d_attn = d_attn
        self.qkv = nn.Linear(d_in, d_attn * 3)
        self.proj = nn.Linear(d_attn, d_out)
        self.softmax = nn.Softmax()

    def forward(self, x):

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, axis=-1)
        w = torch.einsum('bnd, bmd->bnm', q, k)
        a = self.softmax(w * torch.rsqrt(torch.tensor(self.d_attn, dtype=torch.float32)))
        x = torch.einsum('bnm, bmd->bnd', a,v)
        out = self.proj(x)

        return out



class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_queries = x.shape[1]
        num_keys = y.shape[1]
        x = self.query(x)
        y = self.key(y)
        attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        V = self.value(y)
        output = torch.bmm(attn_weights, V)
        
        return output




class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device="cuda"):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        return self.encoding[:x.size(1), :]


class CFPRF_FDN(nn.Module):
    def __init__(self,seq_len=1070,gmlp_layers=1):
        super(CFPRF_FDN, self).__init__()
        self.asr=ASRModel()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.LL=torch.nn.Linear(1024,128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.filts = [128, [1, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128]]
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=self.filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=self.filts[2])),
            nn.Sequential(Residual_block(nb_filts=self.filts[3])),
            nn.Sequential(Residual_block(nb_filts=self.filts[4])),
            nn.Sequential(Residual_block(nb_filts=self.filts[5])),
            nn.Sequential(Residual_block(nb_filts=self.filts[6])))
        self.selu = nn.SELU(inplace=True)
        self.att_dim=[self.filts[-1][-1],64]
        self.attention = nn.Sequential(
            nn.Conv2d(self.att_dim[0], self.att_dim[1], kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(self.att_dim[1]),
            nn.Conv2d(self.att_dim[1], self.att_dim[0], kernel_size=(1,1)),
        )
        self.PE_T = PositionalEncoding(d_model=128, max_len=seq_len)
        self.first_bn1 = nn.BatchNorm2d(num_features=self.filts[-1][-1])
        self.b_amlp=aMLP(d_model=128, d_ffn=-2, seq_len=seq_len, gmlp_layers = gmlp_layers, batch_first=True)
        self.seg_amlp=aMLP(d_model=128, d_ffn=-2, seq_len=seq_len, gmlp_layers = gmlp_layers, batch_first=True)
        self.seg_fc= nn.Linear(128,2)
        self.b_fc= nn.Linear(128,2)
        self.ca=CrossAttention(in_dim=128,out_dim=128)
        self.FFN= FFN(d_model=128, d_ff=128)
        self.norm = nn.LayerNorm(128)
        self.drop = nn.Dropout(0.5, inplace=True)
    def forward(self, x):
        x = self.asr(x)
        x = self.LL(x)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x) 
        x = self.norm(x)
        x = self.selu(x)
        w = self.attention(x) 
        """Difference-aware feature learning module"""
        w_C = F.softmax(w, dim=-1)
        emb_C = torch.sum(x * w_C, dim=-1).transpose(1,2) 
        w_S = F.softmax(w, dim=-3)
        emb_S = torch.sum(x * w_S, dim=-3) 
        emb_T = emb_C + emb_S
        F_DA = emb_T+self.PE_T(emb_T)
        """Boundary-aware feature enhancement module"""
        F_b=self.b_amlp(F_DA)
        F_b=self.drop(F_b)
        bd_scores=self.b_fc(F_b)
        F_CA = self.ca(F_DA, F_b)
        F_CA = self.FFN(F_CA)
        F_BA = self.norm(F_CA + F_DA)
        F_BA=self.seg_amlp(F_BA)
        F_BA=self.drop(F_BA)
        seg_scores=self.seg_fc(F_BA)
        return seg_scores, bd_scores, emb_T, F_BA
