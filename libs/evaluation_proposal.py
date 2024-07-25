
import json
import numpy as np
import torch
import torchvision
import torchaudio
from torch import Tensor
from torch.nn import functional as F, Module
from typing import List, Union


def tiou(proposal, target) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)
    # print(proposal)
    # print(target)
    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union


class AP(torch.nn.Module):
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """
    def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5):
        super().__init__()
        self.iou_thresholds = iou_thresholds
        self.n_labels = 0
        self.ap: dict = {}
    def forward(self, gt_dict, proposals_dict):
        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = 0
            for key in gt_dict.keys():
                proposals = torch.tensor(proposals_dict[key])
                labels = torch.tensor(gt_dict[key])
                self.n_labels += len(labels)
                if len(proposals)==0:
                    continue
                values.append(AP.get_values(iou_threshold, proposals.reshape(-1,3), labels.reshape(-1,2)))
            # sort proposals
            values = torch.cat(values)
            # ind = values[:, 0].sort(stable=True, descending=True).indices
            _,ind= torch.sort(values[:, 0],dim=0,descending=True)
            values = values[ind]
            # accumulate to calculate precision and recall
            curve = self.calculate_curve(values)
            ap = self.calculate_ap(curve)
            self.ap[iou_threshold] = ap

        return self.ap

    def calculate_curve(self, values):
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    @staticmethod
    def get_values(
        iou_threshold: float,
        proposals: Tensor,
        labels: Tensor
    ) -> Tensor:
        n_labels = len(labels)
        n_proposals = len(proposals)
        if n_labels > 0:
            ious = tiou(proposals[:, 1:], labels)
        else:
            ious = torch.zeros((n_proposals, 0))

        # values: (confidence, is_TP) rows
        n_labels = ious.shape[1]
        detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break
        
        is_TP = torch.zeros(n_proposals, dtype=torch.bool)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values


class AR(torch.nn.Module):
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(self, n_proposals_list: Union[List[int], int] = 100, iou_thresholds: List[float] = None,
        parallel: bool = True
    ):
        super().__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_proposals_list: List[int] = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.iou_thresholds = iou_thresholds
        self.parallel = parallel
        self.ar: dict = {}

    def forward(self, gt_dict, proposals_dict):
        for n_proposals in self.n_proposals_list:
            values = torch.zeros((len(proposals_dict), len(self.iou_thresholds), 2))
            self.n_labels = 0
            for idx,key in enumerate(gt_dict.keys()):
                proposals = torch.tensor(proposals_dict[key])
                labels = torch.tensor(gt_dict[key])
                if len(labels)==0: # FP is not considered
                    continue
                values[idx]=AR.get_values(n_proposals,self.iou_thresholds, proposals.reshape(-1,3), labels.reshape(-1,2)) #TP,FN
            values_sum = values.sum(dim=0)
            TP = values_sum[:, 0]
            FN = values_sum[:, 1]
            recall = TP / (TP + FN)
            self.ar[n_proposals] = recall.mean()
        return self.ar

    @staticmethod
    def get_values(
        n_proposals: int,
        iou_thresholds: List[float],
        proposals: Tensor,
        labels: Tensor,
    ):
        proposals = proposals[:n_proposals]
        n_proposals = proposals.shape[0]
        n_labels = len(labels)
        n_thresholds = len(iou_thresholds)
        # values: rows of (TP, FN)
        values = torch.zeros((n_thresholds, 2))
        iou_max=torch.zeros((n_labels))
        
        if n_proposals > 0:
            ious = tiou(proposals[:,1:], labels)
            iou_max = ious.max(dim=0)[0]
            for i in range(n_thresholds):
                iou_threshold = iou_thresholds[i]
                TP = (iou_max > iou_threshold).sum()
                FN = n_labels - TP
                values[i] = torch.tensor((TP, FN))
        else:
            for i in range(n_thresholds):
                TP = 0
                FN = n_labels - TP
                values[i] = torch.tensor((TP, FN))

        return values


