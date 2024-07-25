#!/usr/bin/env python
"""
P2sGrad: 
Zhang, X. et al. P2sgrad: Refined gradients for optimizing deep face models. 
in Proc. CVPR 9906-9914, 2019

https://github.com/nii-yamagishilab/PartialSpoof/blob/main/project-NN-Pytorch-scripts.202102/core_modules/p2sgrad.py
"""
from __future__ import print_function

import numpy as np
import random

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f
from torch.nn import Parameter

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

###################
class P2SActivationLayer(torch_nn.Module):
    """ Output layer that produces cos\theta between activation vector x
    and class vector w_j
    slack: P2SActivationLayer is just the name in the code, but not a 
    layer type like LSTM or pooling.
    It is a fully-connected layer 

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    
    Method: cos\theta = forward(x)
    
    x: (batchsize, input_dim)
    
    cos: (batchsize, output_dim)
          where \theta is the angle between
          input feature vector x[i, :] and weight vector w[j, :]
    """
    def __init__(self, in_planes, out_planes):
        super(P2SActivationLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)

        output:
        -------
          tensor (batchsize, output_dim)
          # for cosine similarity, higher is more similar
          # and score[spoof, bonafide], we use score[:,1] (bonafide score) as final results.
          
        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5).to(input_feat.device)
        
        # normalize the input feature vector
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # done
        return cos_theta

    
class P2SGradLoss(torch_nn.Module):
    """
    P2SGradLoss()
    
    Just MSE loss between output and target one-hot vectors
    """
    def __init__(self, weight=False, label_reverse=False, reverse_ratio = 0.1):
        super(P2SGradLoss, self).__init__()
        self.m_loss = torch_nn.MSELoss()
        self.label_reverse = label_reverse
        self.reverse_ratio = reverse_ratio

    def forward(self, input_score, target):
        """ 
        input
        -----
          input_score: tensor (batchsize, class_num)
                 cos\theta
        output
        ------
          loss: scaler
        """
        # target (batchsize, 1)
        target = target.long() #.view(-1, 1)
        
        # filling in the target
        # index (batchsize, class_num)
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            # index[i][target[i][j]] = 1
            #print(index.shape, target.data.shape)
            index.scatter_(1, target.data.view(-1, 1), 1)

            if(self.label_reverse):
                random.seed(index[:,1].sum().tolist())  # rebuild.
                reverse_select_idx = random.sample(np.arange(index.shape[0]).tolist(),int(index.shape[0]*self.reverse_ratio))
                index[reverse_select_idx] = 1-index[reverse_select_idx]
    
        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(input_score, index)

        return loss

if __name__ == "__main__":
    print("Definition of P2SGrad Loss")
