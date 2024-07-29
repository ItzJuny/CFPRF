import torch
from libs.tool import *
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, det_curve
max_len=340000
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"


def train_one_epoch(train_dlr, model, loss_fun_pgMSE, loss_fun_crl, device, optimizer, rso, v1,v2):
    print("++++++++++++++++++training++++++++++++++++++")
    model.train()
    running_loss = 0
    num_total = 0.0
    step=0
    for batch_x, filenames, batch_seg_label in tqdm(train_dlr,ncols=50):
        if batch_x.shape[1]>max_len:
            continue
        optimizer.zero_grad()
        batch_x, batch_seg_label = batch_x.to(device), batch_seg_label.to(device)
        seg_score, bd_score, emb,_ = model(batch_x)
        # output scores and labels alignmnent
        seg_score, seg_target,seg_score_list,seg_target_list = prepare_segcon_target_ali(batch_seg_label, seg_score,rso)
        seg_target=torch.tensor(seg_target, device=seg_score.device).view(-1).type(torch.long)
        # get boundary labels
        bd_target=seg2bd_label(_seg2bd_label(seg_target)[0],_seg2bd_label(seg_target)[1])
        # boundary scores and labels alignmnent
        bd_target=torch.tensor(bd_target, device=seg_score.device).view(-1).type(torch.long)
        bd_score, bd_target,_,_ = prepare_segcon_target_ali(bd_target.reshape(len(filenames),-1), bd_score, rso)
        # compute loss
        fdl_loss=loss_fun_pgMSE(seg_score, seg_target)
        crl_loss=loss_fun_crl(emb, batch_seg_label)
        fbl_loss=loss_fun_pgMSE(bd_score, bd_target)
        batch_loss =  fdl_loss + v1 * crl_loss+ v2 * fbl_loss
        batch_size = batch_x.size(0)
        running_loss += (batch_loss.item() * batch_size)
        batch_loss.backward()
        optimizer.step()
        num_total+=batch_size
    running_loss /= num_total
    return running_loss
        

def test_one_epoch(infer_dlr, gt_dict, model, rso, device):
    print("++++++++++++++++++testing++++++++++++++++++")
    model.eval()
    with torch.no_grad():
        seg_score_dict={}
        seg_tar_dict={}
        cp_dict={}
        for batch_x, filenames, batch_seg_label in tqdm(infer_dlr,ncols=50):
            if batch_x.shape[1]>max_len:
                for idx,fn in enumerate(np.array(filenames)):
                    cp_dict[fn]=np.array([])
                continue
            batch_x, batch_seg_label = batch_x.to(device), batch_seg_label.to(device)
            seg_score,_, _,_ = model(batch_x) 
            seg_score, seg_target,seg_score_list,seg_target_list = prepare_segcon_target_ali(batch_seg_label, seg_score,rso)
            seg_score_np=np.array([ss.data.cpu().numpy() for ss in seg_score_list])
            cp_list=[segscore2proposal(seg_score_np[idx],cp_fun=proposal_func,rso=rso)[1] for idx in range(len(batch_x))]
            for idx,fn in enumerate(np.array(filenames)):
                cp_dict[fn]=frame2second_proposal(cp_list[idx], rso=rso)
                seg_score_dict[fn]=seg_score_np[idx][:,1].ravel()
                seg_tar_dict[fn]=seg_target_list[idx].data.cpu().numpy().ravel()
        EER,_,_,_,_,_=eval_PFD(seg_score_dict,seg_tar_dict)
        cp_mAP,_,_=eval_TFL(gt_dict,cp_dict, detail=True)
    return EER, cp_mAP
      

def _seg2bd_label(seglabel):
    
    """
        transfer binary label to 'T'/'F' and during lengths.
        authenticity: 1110001111111
        authenticity: TTTFFFTTTTTTT
        labels, lengths: TFT, 337
    """
    seglabel=seglabel.data.cpu().numpy()
    labels = []
    lengths = []
    current_length = 1  
    current_label = 'T' if seglabel[0] == 1 else 'F'  
    for i in range(1, len(seglabel)):
        if seglabel[i] == seglabel[i-1]:  
            current_length += 1 
        else:
            labels.append(current_label)
            lengths.append(current_length)
            current_label = 'T' if seglabel[i] == 1 else 'F'
            current_length = 1
    labels.append(current_label)
    lengths.append(current_length)
    return labels, lengths


def seg2bd_label(labels, lengths):
    """
        get the boundary labels based on the paper titled 
        "Integrating frame-level boundary detection and 
        deepfake detection for locating manipulated regions 
        in partially spoofed audio forgery attacks"

        authenticity: 1110001111111
        authenticity: TTTFFFTTTTTTT
        labels, lengths: TFT, 337
        ->boundary:    0011011000000
        
    """
    res=[]
    for idx in range(len(labels)):
        label, length=labels[idx], lengths[idx]
        temp=np.zeros((length))
        temp[0]=1
        temp[-1]=1
        if idx==0 and label=='T':
            temp[0]=0
        if idx==len(labels)-1 and label=='T' and length>1:
            temp[-1]=0
        res.extend(temp)
    return np.array(res).reshape(-1,1)
