import sys
sys.path.append("..")
from libs.tool import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import numpy as np
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
        self.mp = nn.MaxPool2d((1, 3))  

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
        out = self.mp(out)
        return out




class TwoMLPHead(nn.Module):
    def __init__(self, in_channels=49, representation_size=256):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class Predictor(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super(Predictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return torch.sigmoid(scores), bbox_deltas


class CFPRF_PRN(nn.Module):
    def __init__(self,device):
        super(CFPRF_PRN, self).__init__()
        self.encoder=TwoMLPHead()
        self.headers=Predictor() 
        self.device=device
        self.output_size=7
        self.cnn1=Residual_block(nb_filts=[1,32], first=True)
        self.cnn2=Residual_block(nb_filts=[32,64], first=True)
        
    def boundary(self,seg_feature):
        coarse_proposal_list=[]
        for idxx, seg_f in enumerate(seg_feature):
            pred_label=np.argmax(seg_f.data.cpu().numpy(), axis=1)
            fake_start,fake_end,_,_ = rso2time(np.array(pred_label,dtype=int).reshape(-1))
            proposal=gt_proposal(fake_start,fake_end)
            coarse_proposal_list.append(proposal)
        return coarse_proposal_list
    def addbatch_list2np(self,cp_list):
        """ Add the batch_idx indicator to the coarse-grained proposals [c,s,e], [b_idx, c, s,e]
        Args:
            cp_list (list): 
            [array([[  1.,   0.,  37.],
                [  1.,  75.,  76.],
                [  1., 113., 114.],
                [  1., 117., 120.]]), 
            array([[  1.,   0.,   2.],
                [  1.,  12.,  13.],
                [  1.,  30., 120.]])]
        Returns:
            batch_cp_np (numpy): 
            [[  0.   1.   0.  37.]
            [  0.   1.  75.  76.]
            [  0.   1. 113. 114.]
            [  0.   1. 117. 120.]
            [  1.   1.   0.   2.]
            [  1.   1.  12.  13.]
            [  1.   1.  30. 120.]
        """
        batch_size = len(cp_list)
        batch_cp_list=[np.insert(pro,0,b_idx) for b_idx in range(batch_size) for pro in cp_list[b_idx]]
        batch_cp_np=np.array(batch_cp_list)
        return batch_cp_np
    
    def rebatch_np2list(self, cp_np_cpu, batch_size):
        """ Remove the batch_idx indicator and return the proposal list
        Args:
            cp_np_cpu (numpy): 
                [[  0.   0.95   0.  37.]
                [  0.   0.11  75.  76.]
                [  0.   0.5 113. 114.]
                [  0.   0.31 117. 120.]
                [  1.   0.33   0.   2.]
                [  1.   0.1  12.  13.]
                [  1.   0.96  30. 120.]
            batch_size (int),
        Returns:
            split_list (list): 
                [array([[  0.95,   0.,  37.],
                    [  0.11,  75.,  76.],
                    [  0.5, 113., 114.],
                    [  0.31, 117., 120.]]), 
                array([[  0.33,   0.,   2.],
                    [  0.1,  12.,  13.],
                    [  0.96,  30., 120.]])]
        """
        unique_batches = np.unique(cp_np_cpu[:, 0])
        split_list = []
        for batch_ix in unique_batches:
            batch_rows = cp_np_cpu[cp_np_cpu[:, 0] == batch_ix]
            batch_rows_no_ix = np.delete(batch_rows, 0, axis=1)
            split_list.append(batch_rows_no_ix)
        for batch_ix in range(batch_size):
            if batch_ix not in unique_batches:
                split_list.insert(batch_ix, np.array([]))
        return split_list
    
    def forward(self, embs, cp_list, rso):
        batch_cp_np=self.addbatch_list2np(cp_list)
        shift=int(rso/20)
        extracted_rois=[]
        for proposal in batch_cp_np:
            batch_ix, _, start, end = proposal
            batch_ix = int(batch_ix)
            start, end = int(start * shift), int(end*shift)
            roi_region = torch.tensor(embs[batch_ix, start:end, :], dtype=torch.float32).to(self.device)
            roi_region = roi_region.unsqueeze(0).unsqueeze(0) 
            roi_region=self.cnn1(roi_region)
            roi_region=self.cnn2(roi_region)
            roi_region=roi_region.permute(0,2,1,3)
            roi_features = F.adaptive_max_pool2d(roi_region, (self.output_size,self.output_size))
            extracted_rois.append(torch.sum(roi_features,dim=1))
        concatenated_rois = torch.cat(extracted_rois, dim=0)
        o = self.encoder(concatenated_rois)
        ver_score, reg_predict = self.headers(o)
        # add confidence score to proposals
        PRN_cp_np = np.copy(batch_cp_np)
        PRN_cp_np[:, 1] = ver_score.data.cpu().numpy().reshape(-1) 
        # remove the batch_idx indicator
        PRN_cp_list = self.rebatch_np2list(PRN_cp_np,len(cp_list))
        return ver_score, reg_predict, PRN_cp_np, PRN_cp_list

