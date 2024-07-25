
import numpy as np
import torch,sys
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, det_curve,confusion_matrix,auc,roc_curve
import time
import librosa
from .evaluation_proposal import *
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"



# For evaluation

def eval_PFD(seg_score_dict,seg_tar_dict):
    """
        for partial forgery detection evaluation 
    Args:
        seg_score_dict (dict): predicted segmental scores
            {
                filename_1:[0.98,0.68,0.1,0.50,0.1,0.0],
                filename_2:[0.0,0.11,0.51,0.60,0.1,0.0],
                ...
            }
        
        seg_tar_dict (dict): true segmental labels
            {
                filename_1:[0,1,1,1,1,1],
                filename_2:[0,0,0,0,1,0],
                ...
            }
        
    Returns:
        EER, ACC, F1, PRECISION, RECALL, AUC
    """
    label_np, score_np = dict2np(seg_tar_dict), dict2np(seg_score_dict)
    """---------EER----------"""
    frr, far, thresholds = det_curve(label_np, score_np)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    print("EER=%.2f%%"%(eer*100))
    EER_threshold=thresholds[min_index]
    ACC_threshold=EER_threshold
    """---------Others----------"""
    pred_label_dict={}
    for key in seg_score_dict.keys():
        pred_label_dict[key]=pred_label(seg_score_dict[key],ACC_threshold)
    pred_label_np=dict2np(pred_label_dict)
    f1=f1_score(label_np, pred_label_np)*100
    pre=precision_score(label_np, pred_label_np)*100
    rec=recall_score(label_np, pred_label_np)*100
    acc=accuracy_score(label_np, pred_label_np)*100
    fpr,tpr,ths=roc_curve(label_np, score_np)
    AUC=auc(fpr, tpr)*100 
    print("F1",f1, "precision", pre, "recall", rec, "ACC", acc, "auc", (AUC))
    return eer*100,acc,f1,pre,rec, AUC


def eval_TFL(groundtrue_dict, coarse_proposal_dict, detail=False):
    
    """
        for temporal forgery localization evaluation 
    Args:
        coarse_proposal_dict (dict): coarse-grained proposals
            {
                filename_1:{
                    [confidence_1, start_second_1, end_second_1],
                    [confidence_2, start_second_2, end_second_2],
                    ...
                        },
                filename_2:{
                    [confidence_1, start_second_1, end_second_1],
                    [confidence_2, start_second_2, end_second_2],
                    ...
                }
                ...
            }
        
        groundtrue_dict (dict): groundtruths
            {
                filename_1:{
                    [start_second_1, end_second_1],
                    [start_second_2, end_second_2],
                    ...
                        },
                filename_2:{
                    [start_second_1, end_second_1],
                    [start_second_2, end_second_2],
                    ...
                }
                ...
            }
        
    Returns:
        mAP_value, ap_score, ar_score
    """
    iou_thresholds = [0.5, 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9, 0.95]
    n_proposals_list = [50, 20, 10, 5, 2, 1]
    ap_score = AP(iou_thresholds=iou_thresholds)(groundtrue_dict,coarse_proposal_dict)
    mAP_value=np.array(list(ap_score.values())).mean()
    if detail:
        ar_score = AR(n_proposals_list, iou_thresholds=iou_thresholds)(groundtrue_dict,coarse_proposal_dict)
        print("AP@0.5=%.3f,AP@0.75=%.3f,AP@0.95=%.3f,mAP=%.3f"%(ap_score[0.5],ap_score[0.75],ap_score[0.95],mAP_value))
        print("AR@2=%.3f,AR@5=%.3f,AR@10=%.3f,AR@20=%.3f,AR@50=%.3f"%(ar_score[2],ar_score[5],ar_score[10],ar_score[20],ar_score[50]))
        return mAP_value, ap_score, ar_score
    else:
        print("mAP=%.3f"%(mAP_value))
        return mAP_value


def dict2np(item):
    """
    Args:
        item (dict): 
            {
                filename_1:[0.98,0.68,0.1,0.50,0.1,0.0],
                filename_2:[0.0,0.11,0.51,0.60,0.1,0.0],
            }
    Returns:
        dict_np(numpy): flatten array
            array([0.98,0.68,0.1,0.50,0.1,0.0, 0.0,0.11,0.51,0.60,0.1,0.0, ...])
    """
    dict_np=[]
    for key in item.keys():
        dict_np.extend(item[key])
    np.array(dict_np)
    return dict_np


def pred_label(scores,thr):
    """
        if score>threshold for score in scores
        then predict label is set to 1 elsewise 0
    """
    pred_label_list=[]
    for i in scores:
        if i>=thr:
            pred_label_list.append(1)
        else:
            pred_label_list.append(0)
    return np.array(pred_label_list)


# process the FDN output


def vec_in_scale(vec, rso):
    """ 
        downsample the vector to different time resolutions
    """
    if rso==20:
        return vec
    else:
        shift=int(rso/20) # if rso=40ms
        num_frames = int(len(vec) / shift) #T'=T/2
        new_vec = torch.zeros((num_frames,2), dtype=vec.dtype)
        for idx in range(num_frames):
            st, et  = int(idx * shift), int((idx+1)*shift)
            if et > len(vec):
                et = len(vec)
            new_vec[idx, 0] = torch.mean(vec[st:et, 0])
            new_vec[idx, 1] = torch.mean(vec[st:et, 1])
        return new_vec

def prepare_segcon_target_ali(seg_targets, seg_vecs, rso=20):
    """
        downsample input to different time resolutions and align the length between true labels and predicted segmental-level scores

    Args:
        seg_targets (tensor): true labels
        seg_vecs (tensor): predicted scores
        rso (int, optional): downsampled time resolution. Defaults to 20.

    Returns:
        align_seg_vec (tensor):
        align_seg_target (tensor):
        align_seg_vec (list):
        align_seg_target (list):
    """
    align_seg_vec=[]
    align_seg_target=[]
    for idx, (seg_label, seg_score) in enumerate(zip(seg_targets, seg_vecs)):
        seg_score=vec_in_scale(seg_score, rso)
        min_len=min(seg_label.shape[0],seg_score.shape[0])
        align_seg_vec.append(seg_score[:min_len])
        align_seg_target.append(seg_label[:min_len])
    return torch.cat(align_seg_vec), torch.cat(align_seg_target),align_seg_vec,align_seg_target



# coarse-gained proposal

def proposal_func(fake_start, fake_end, rso=20):
    """function to produce proposals

    Args:
        fake_start (numpy): forgery start frames
        fake_end (numpy):  forgery end frames
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        regions (list): list of forgery proposals
        [[start_frame, end_frame], ...]
    """
    regions=[]
    for i in range(len(fake_start)):
        regions.append([fake_start[i]/rso, fake_end[i]/rso])
    return regions

def segscore2proposal(segscore_2d,cp_fun,rso=20):
    """ 
        segmental scores to temporal proposals

    Args:
        segscore_2d (numpy): 2D segmental scores produced by FDN
        cp_fun:  function to produce proposals
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        pred_label (numpy): using segmental scores via the np.argmax function to produce predicted label
        proposals_list (numpy): forgery proposals with the initial confidence score set to 1
        [[1, start_frame, end_frame], ...]
    """
    pred_label=np.array([np.argmax(segscore_2d, axis=1)],dtype=float).reshape(-1)
    proposals=cp_fun(_seglabel2proposal(pred_label, rso)[1],_seglabel2proposal(pred_label, rso)[2],rso)
    proposals_list=[np.insert(pro,0,1) for pro in proposals]
    return pred_label,np.array(proposals_list).reshape(-1,3)


def _seglabel2proposal(rsolabel, rso=20):
    """
        segmental labels to temporal proposals
    Args:
        rsolabel (numpy): segmental labels
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        fake_segments (numpy): forgery segments
        starts (numpy): forgery start frames
        ends (numpy): forgery end frames
    """
    fake_segments = []
    prev_label = None
    current_start = 0
    fake_label=0
    true_label=1
    for i, label in enumerate(rsolabel):
        label = int(label)
        time = i * rso
        # Detect state changes for fake segments only
        if label == fake_label and prev_label == true_label: # mark fake start
            current_start = time
            if i == len(rsolabel) - 1:  # the end
                fake_segments.append((current_start, time + rso))
        elif label == true_label and prev_label == fake_label:# mark fake end
            fake_segments.append((current_start, time))
        elif label == fake_label and i == len(rsolabel) - 1: # the end
            fake_segments.append((current_start, time + rso))
        prev_label = label
    fake_segments = np.array([[float(start), float(end)] for start, end in fake_segments],dtype=float).reshape(-1,2)
    starts=fake_segments[:,0]
    ends=fake_segments[:,1]
    return fake_segments, starts, ends


def get_gt_list(batch_seg_label,rso=20):
    """
        segmental labels to groudtruth

    Args:
        batch_seg_label (tensor): labels
        rso (int, optional): time resolution. Defaults to 20.

    Returns:
        gt_list (list): list of groudtruth
    """
    gt_list=[]
    for seg_label in batch_seg_label.data.cpu().numpy():
        _, starts, ends = _seglabel2proposal(seg_label, rso)
        gt_list.append(proposal_func(starts, ends, rso))
    return gt_list
 
# VerH


def get_extend_cplist(ver_label_list, cp_list, gt_list, T_length):
    """
        random sample pos. or neg. proposals based on groudtruths

    Args:
        ver_label_list (list): list of VerH labels
        cp_list (list): list of coarse-grained proposals
        gt_list (list): list of grouhgtruths
        T_length (int): length of segments

    Returns:
        extend_cp_list (list): the extended list of coarse-grained proposals
        extend_ver_label_list (list): the extended list of VerH labels
        extend_matched_gt_list (list): the extended list of matched grouhgtruths
        
    """
    extend_cp_list, extend_matched_gt_list,extend_ver_label_list = [], [], []
    for ver_label_idx, cp_idx, gt_idx in zip(ver_label_list, cp_list, gt_list):
        p_num = np.sum(ver_label_idx == 1)
        n_num = np.sum(ver_label_idx == 0)
        if p_num <= n_num or np.array(gt_idx).size==0:
            extend_cp_list.append(np.array([]))
            extend_ver_label_list.append(np.array([]))
            extend_matched_gt_list.append(np.array([]))
        elif p_num > n_num:
            flag=0
            gen_num = abs(p_num - n_num)
            extend_label = [flag] * gen_num
            # balance the number of neg. and pos. proposals
            extend_cp, extend_matchgt = generate_unique_pairs(flag, T_length, gt_idx, cp_idx, gen_num)
            extend_cp_list.append(np.array(extend_cp).reshape(-1,3))
            extend_ver_label_list.append(np.array(extend_label).reshape(-1))
            extend_matched_gt_list.append(np.array(extend_matchgt).reshape(-1,2))
    return extend_cp_list, extend_ver_label_list, extend_matched_gt_list

def generate_unique_pairs(flag, T_length, gt_idx, cp_idx, gen_num):
    """
        generated proposals 

    Args:
        flag (int): indicator to generate pos. or neg. proposals
        T_length (int): length of segments
        gt_idx (_type_): index of groundtruths
        cp_idx (_type_): index of coarse-grained proposals
        gen_num (_type_): number to generate proposals

    Returns:
        generated_pairs (_type_):
        generated_mathgts (_type_):
    """
    import random
    generated_pairs, generated_mathgts = [], []
    while len(generated_pairs) < gen_num:
        if flag==1:
            idx_rand = random.randint(0, len(gt_idx) - 1)
            S = random.randint(gt_idx[idx_rand][0]-3, gt_idx[idx_rand][0] +3)
            E = random.randint(gt_idx[idx_rand][1]-3, gt_idx[idx_rand][1] +3)
            if S<0:
                S=0
            if E>=T_length:
                E=T_length-1
        else:
            S = random.randint(0, T_length - 1)
            E = random.randint(S + 1, T_length)
        # make sure new_pair is unique
        new_pair = np.array([flag, S, E]).reshape(1,3)
        match_flag, match_gt = verhead_label(new_pair, gt_idx)
        if np.array(match_flag)==flag:
            if np.array(verhead_label(new_pair, cp_idx[:, 1:])[0]) == 0: 
                if not generated_pairs or np.array(verhead_label(new_pair, np.array(generated_pairs).reshape(-1, 3)[:, 1:])[0]) == 0:
                    generated_pairs.append(new_pair)
                    generated_mathgts.append(match_gt)
    return generated_pairs, generated_mathgts

def frame2second_proposal(proposal_list, rso=20):
    """
    Args:
        [confidence score, start_frame, end_end]
    output:
        [confidence score, start_second, end_second]
    """
    return [[a, b *rso/ 1000, c * rso/ 1000] for a, b, c in proposal_list]

def vstack_twolist(list1, list2):
    list1_array = np.array(list1)
    list2_array = np.array(list2)
    if list1_array.size == 0:
        result = list2_array
    elif list2_array.size == 0:
        result = list1_array
    else:
        result = np.vstack((list1_array, list2_array))
    return result


# decoder


def verhead_label(cp_list, gt_list):
    """
        get the VerH labels and matched grouhgtruths

    Args:
        cp_list (list): list of coarse-grained proposals
        gt_list (list): list of grouhgtruths

    Returns:
        ver_labels (list): list of VerH labels
        matched_gt (list): list of matched grouhgtruths
        
    """
    ver_labels = []
    matched_gt = []
    for cp_idx, gt_idx in zip(cp_list, gt_list):
        cp_idx, gt_idx=np.array(cp_idx),np.array(gt_idx).reshape(-1,2)
        if len(cp_idx) == 0 or len(gt_idx) == 0:
            ver_labels.append(np.zeros(len(cp_idx), dtype=float) if len(cp_idx) > 0 else np.array([]))
            matched_gt.append(np.zeros((len(cp_idx), 2), dtype=int) if len(cp_idx) > 0 else np.array([]))
            continue
        match_quality_matrix = tiou(np.array(gt_idx).reshape(-1, 2), np.array(cp_idx).reshape(-1, 3)[:, 1:])
        matched_vals, matches = match_quality_matrix.max(dim=0)
        matches[matched_vals < 0.5] = -1 
        matches_idx = matches >= 0
        ver_labels.append(matches_idx.numpy().astype(float))
        matched_gt.append(gt_idx[matches.clamp(min=0).numpy()])
    return ver_labels, matched_gt



def encoder_reg(proposal, gt, scales=[100,100]):
    """
        compute the regression labels
    Args:
        proposal (numpy): [confidence_1, start_1, end_1]...
        gt (list): [start_1, end_1]...
        scales (list, optional): scale the regression labels. Defaults to [100,100].

    Returns:
        list:  [shift_1, dur_1]...
        dur=np.log(gt_dur / proposal_dur)* scales[1]
        shift = ((gt_start - proposal_start) / proposal_dur) * scales[0] 
    """
    proposal, gt = np.array(proposal).reshape(-1,3), np.array(gt).reshape(-1,2)
    gt_dur=np.array(gt[:,1]-gt[:,0]).reshape(-1,1)
    proposal_dur=np.array(proposal[:,2]-proposal[:,1]).reshape(-1,1)
    gt_start=np.array(gt[:,0]).reshape(-1,1)
    proposal_start=np.array(proposal[:,1]).reshape(-1,1)
    reg_dur = np.log(gt_dur / proposal_dur)* scales[1] 
    reg_shift = ((gt_start - proposal_start) / proposal_dur) * scales[0] 
    reg = np.concatenate((reg_shift, reg_dur), axis=1)
    return reg

def decoder_reg(proposal,reg,scales=[100,100]):
    """
        apply the predicted offsets to the proposals

    Args:
        proposal (numpy): coarse-grained proposals [confidence_1, start_1, end_1],...
        reg (list): regression offsets [shift_1, dur_1],...
        scales (list, optional): scale the regression offsets. Defaults to [100,100].

    Returns:
        fine-grained proposal (numpy): [confidence_1, start_1, end_1]
    """
    proposal,reg=np.array(proposal).reshape(-1,3), np.array(reg).reshape(-1,2)
    score=np.array(proposal[:,0]).reshape(-1,1)
    proposal_start=np.array(proposal[:,1]).reshape(-1,1)
    proposal_end=np.array(proposal[:,2]).reshape(-1,1)
    reg_shift, reg_dur=np.array(reg[:,0]).reshape(-1,1), np.array(reg[:,1]).reshape(-1,1)
    proposal_dur=proposal_end-proposal_start
    refine_dur=np.exp(reg_dur/scales[1])*(proposal_dur) 
    refine_start=((reg_shift/scales[0])*proposal_dur)+proposal_start 
    refine_end=refine_start+refine_dur
    #refine_start
    refine_start[refine_start < 0] = 0
    refine_end[refine_end < 0] = 0
    return np.concatenate((score, refine_start, refine_end), axis=1)



def decoder_ver(cp_list, ver_label_pred):
    """
        approach positive proposals to groundtruths.

    Args:
        cp_list (list): list of coarse-grained proposals
        ver_label_pred (list): list of VerH prediction labels

    Returns:
        cp_ver_list (list): list of positive coarse-grained proposals using VerH prediction labels
    """
    indices = np.cumsum([arr.shape[0] for arr in cp_list])  
    ver_label_parts = np.split(ver_label_pred.flatten(), indices[:-1]) 
    cp_ver_list = [arr[ver_label_part == 1] for arr, ver_label_part in zip(cp_list, ver_label_parts)]
    return cp_ver_list

def decoder_ver2(fp_flatten, pos_label, ref_cp_list):
    """
        1. Shape recovery: fp_flatten->fp_list by ref_cp_list.shape
        2. VerH Decoder using pos_label
    Args:
        fp_flatten (_type_): _description_
        pos_label (_type_): predicted by VerH
        ref_cp_list (list): shape reference for recovery

    Returns:
        _type_: _description_
    """
    fp_list=flatten2list(fp_flatten, ref_cp_list)
    return decoder_ver(fp_list, pos_label)


def flatten2list(flat_array, ref_cp_list):
    """
        Shape recovery from array to list by ref_cp_list.shape

    Args:
        flat_array (numpy): flatten array
        ref_cp_list (list): shape reference for recovery

    """
    shapes = [arr.shape for arr in ref_cp_list]
    proposal_list = []
    start = 0
    for shape in shapes:
        if shape[0] != 0: 
            size = shape[0] 
            end = start + size
            proposal_list.append(flat_array[start:end].reshape(-1, shape[-1]))
            start = end
        else:
            proposal_list.append(np.array([]).reshape(0, 3)) 
    return proposal_list

# npy

def writenpy(filepath,content):
    np.save(filepath, content)
    

def readnpy(filepath):
    return np.load(filepath,allow_pickle=True).item()
