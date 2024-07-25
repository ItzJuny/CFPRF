import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np
"""
Modified from:
    Authors: Xie Y, Cheng H, Wang Y, et al. 
    An Efficient Temporary Deepfake Location Approach Based Embeddings for Partially Spoofed Audio Detection.
    in Proc. ICASSP, 2024: 966-970.
"""

class CRLoss(nn.Module):
    def __init__(self):
        super(CRLoss, self).__init__()
        self.th_similar_min = 0.9
        self.th_different_max = 0.1

    def cosine_similarity(self, x1, x2, eps=1e-8):
        '''
        pair-wise cosine distance
        x1: [M, D]
        x2: [N, D]
        similarity: [M, N]
        '''
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        similarity = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return similarity

    def forward(self, embeddings, label):
        loss_batch = 0
        num_batch = embeddings.size()[0]
        num_batch_dynamic = num_batch
        # print("num_batch",num_batch)
        for ibat in range(num_batch):
            # embedding = embeddings[ibat]  # [T, 128]
            # print(len(label[ibat]), embeddings[ibat].shape[0])
            min_length = min(len(label[ibat]), embeddings[ibat].shape[0])
            label_ibat = label[ibat, :min_length]
            embedding = embeddings[ibat, :min_length]  # obtain the true length before zero padding
            # obtain the true length before zero padding
            real_mask = torch.where(label_ibat == 1)[0]  # real frame position
            fake_mask = torch.where(label_ibat == 0)[0]  # fake frame position
            # print(real_mask,)
            # print(fake_mask,)
            Real_embedding = embedding[real_mask]  # [K, 128]
            Fake_embedding = embedding[fake_mask]  # [L, 128]
            r_embedding = Real_embedding
            f_embedding = Fake_embedding
            if Real_embedding.size()[0] == 0:  # if no real frames, all fake embeddings should be similar
                sim_f2f = self.cosine_similarity(f_embedding, f_embedding)
                sim_f2f_hard = torch.min(sim_f2f, dim=1)[0]
                zero = torch.zeros_like(sim_f2f_hard)
                loss_f2f = torch.max(self.th_similar_min - sim_f2f_hard, zero)
                loss_f2f = loss_f2f.mean()
                continue
            if Fake_embedding.size()[0] == 0:  # if no fake frames, all real embeddings should be similar
                sim_r2r = self.cosine_similarity(r_embedding, r_embedding)
                sim_r2r_hard = torch.min(sim_r2r, dim=1)[0]
                zero = torch.zeros_like(sim_r2r_hard)
                loss_r2r = torch.max(self.th_similar_min - sim_r2r_hard, zero)
                loss_r2r = loss_r2r.mean()
                continue
            # all fake embedings should be similar
            sim_f2f = self.cosine_similarity(f_embedding, f_embedding)
            sim_f2f_hard = torch.min(sim_f2f, dim=1)[0]
            zero = torch.zeros_like(sim_f2f_hard)
            loss_f2f = torch.max(self.th_similar_min - sim_f2f_hard, zero)
            loss_f2f = loss_f2f.mean()

            # all real embeddings should be similar
            sim_r2r = self.cosine_similarity(r_embedding, r_embedding)
            sim_r2r_hard = torch.min(sim_r2r, dim=1)[0]
            zero = torch.zeros_like(sim_r2r_hard)
            loss_r2r = torch.max(self.th_similar_min - sim_r2r_hard, zero)
            loss_r2r = loss_r2r.mean()

            # fake embeddings should be different with real embeddings
            sim_f2r = self.cosine_similarity(f_embedding, r_embedding)
            # f2r
            sim_f2r_hard = torch.max(sim_f2r, dim=1)[0]
            zero = torch.zeros_like(sim_f2r_hard)
            loss_f2r = torch.max(sim_f2r_hard - self.th_different_max, zero)
            loss_f2r = loss_f2r.mean()
            # r2f
            sim_r2f = self.cosine_similarity(r_embedding, f_embedding)
            sim_r2f_hard = torch.max(sim_r2f, dim=1)[0]
            zero = torch.zeros_like(sim_r2f_hard)
            loss_r2f = torch.max(sim_r2f_hard - self.th_different_max, zero)
            loss_r2f = loss_r2f.mean()
            
            loss_batch = loss_batch + loss_f2f + loss_r2r + loss_f2r + loss_r2f

        loss_batch = loss_batch / num_batch_dynamic
        return loss_batch
