import sys
from libs.tool import *
from tqdm import tqdm
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, det_curve
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"


def train_one_epoch(train_dlr, FDN_model, RPN_model, loss_ver, rso, device, optimizer):
    print("++++++++++++++++++training++++++++++++++++++")
    RPN_model.train()
    FDN_model.eval()
    running_loss = 0
    num_total = 0.0
    ver_pred_labels=[]
    ver_true_labels=[]
    regs_true_pos=[]
    regs_pred_pos=[]
    for batch_x, filenames,batch_seg_label_np in tqdm(train_dlr,ncols=50):
        optimizer.zero_grad()
        batch_x, batch_seg_label = batch_x.to(device), batch_seg_label_np.to(device)
        batch_size = batch_x.size(0)
        # inference FDN
        with torch.no_grad():
            seg_score,_, embs, last_hidden_states = FDN_model(batch_x) 
            gt_list = get_gt_list(batch_seg_label, rso=rso)
            seg_score, seg_target, seg_score_list,seg_target_list = prepare_segcon_target_ali(batch_seg_label, seg_score,rso)
            seg_score_np=np.array([ss.data.cpu().numpy() for ss in seg_score_list])
            # get coarse-gained proposal lists from FDN output scores with the initial confidence score set to 1
            FDN_cp_list=[segscore2proposal(seg_score_np[idx],cp_fun=proposal_func,rso=rso)[1] for idx in range(batch_size)]
            if all(arr.size == 0 for arr in FDN_cp_list):
                # if the coarse proposal list is null, continue
                continue
            # random sample negative proposals based on groudtruths to balance 
            # the number of negative and positive proposals for training VerH, 
            # which is not needed during the testing process.
            ver_label_list, matched_gt_list= verhead_label(FDN_cp_list, gt_list)
            extend_cp_list, extend_ver_label_list, extend_matched_gt_list= get_extend_cplist(ver_label_list, FDN_cp_list, gt_list, T_length=seg_score_np.shape[1])
            BALANCE_ver_label=[np.hstack((np.array(ver_label_list[ii]),np.array(extend_ver_label_list[ii]))) for ii in range(batch_size)]
            BALANCE_cp_list=[vstack_twolist(FDN_cp_list[ii],extend_cp_list[ii]) for ii in range(batch_size)]
            BALANCE_matched_gt_list=[vstack_twolist(matched_gt_list[ii],extend_matched_gt_list[ii]) for ii in range(batch_size)]
        # training FDN
        ver_pred, reg_pred, _, PRN_cp_list = RPN_model(last_hidden_states, BALANCE_cp_list, rso)
        ver_label_pred= (ver_pred > 0.5).int()
        # compute the VerH loss
        BALANCE_ver_label_flatten = np.concatenate(BALANCE_ver_label).reshape(-1,1)
        BALANCE_ver_label = torch.tensor(BALANCE_ver_label_flatten).to(device)
        ver_true_labels.extend(BALANCE_ver_label.cpu().numpy())
        ver_pred_labels.extend(ver_label_pred.cpu().numpy())
        ver_loss = loss_ver(ver_pred, BALANCE_ver_label.float())
        # compute the RegH loss 
        # using 'BALANCE_ver_label_flatten' instead of 'ver_label_pred' for training
        sampled_pos_inds = np.where(BALANCE_ver_label_flatten == 1)[0]   
        PRN_cp_flatten = np.concatenate([arr for arr in PRN_cp_list if arr.size > 0]).reshape(-1, 3)
        BALANCE_mgt_flatten = np.concatenate([arr for arr in BALANCE_matched_gt_list if arr.size > 0]).reshape(-1, 2)
        # forgery length less than 2 frames is not considered
        lengths = BALANCE_mgt_flatten[sampled_pos_inds][:, 1] - BALANCE_mgt_flatten[sampled_pos_inds][:, 0]
        short_inds = np.where(lengths < 2)[0]
        sampled_pos_inds = np.delete(sampled_pos_inds, short_inds)
        if sampled_pos_inds.size >0: # regression is performed only in the case of having positive samples
            reg_trues = encoder_reg(PRN_cp_flatten[sampled_pos_inds], BALANCE_mgt_flatten[sampled_pos_inds])
            sampled_pos_inds = torch.tensor(sampled_pos_inds).to(device)
            reg_trues = torch.tensor(reg_trues).to(device)
            reg_loss = F.smooth_l1_loss(reg_pred[sampled_pos_inds].float().to(device),torch.tensor(reg_trues).float().to(device),reduction="sum",) / (sampled_pos_inds.numel())
            regs_true_pos.extend(reg_trues.data.cpu().numpy())
            regs_pred_pos.extend(reg_pred[sampled_pos_inds].data.cpu().numpy())
            batch_loss =  ver_loss+0.2*reg_loss 
        else:
            batch_loss =  ver_loss
        running_loss += (batch_loss.item() * batch_size)
        batch_loss.backward()
        optimizer.step()
        num_total+=batch_size
    running_loss /= num_total
    acc=accuracy_score(np.array(ver_true_labels).reshape(-1),np.array(ver_pred_labels).reshape(-1))*100
    print("VerH, ACC=",acc,"Running_loss=",running_loss)
    mse_shift=metrics.mean_squared_error(np.array(regs_true_pos)[:,0], np.array(regs_pred_pos)[:,0])
    mse_dur=metrics.mean_squared_error(np.array(regs_true_pos)[:,1], np.array(regs_pred_pos)[:,1])
    print("RerH, MSE: Shift={:.4f} Dur={:.4f}".format(mse_shift, mse_dur))
    return running_loss,acc


def test_one_epoch(eval_dlr, test_gt_dict, FDN_model, RPN_model, rso, device):
    print("++++++++++++++++++testing++++++++++++++++++")
    RPN_model.eval()
    FDN_model.eval()
    with torch.no_grad():
        cp_dict={}
        ver_dict={}
        regver_dict={}
        seg_score_dict={}
        seg_tar_dict={}
        for batch_x, filenames, batch_seg_label in tqdm(eval_dlr,ncols=50):
            for fn in filenames:
                cp_dict[fn]=np.array([])
                ver_dict[fn]=np.array([])
                regver_dict[fn]=np.array([])
            if batch_x.shape[1]>340000:
                continue
            batch_x, batch_seg_label = batch_x.to(device), batch_seg_label.to(device)
            batch_size = batch_x.size(0)
            # inference FDN
            seg_score,_, embs, last_hidden_states = FDN_model(batch_x) 
            seg_score, seg_target, seg_score_list,seg_target_list = prepare_segcon_target_ali(batch_seg_label, seg_score,rso)
            seg_score_np=np.array([ss.data.cpu().numpy() for ss in seg_score_list])
            # get coarse-gained proposal lists from FDN output scores with the initial confidence score set to 1
            FDN_cp_list=[segscore2proposal(seg_score_np[idx],cp_fun=proposal_func,rso=rso)[1] for idx in range(batch_size)]
            # save dict for FDN output 
            for idx,fn in enumerate(np.array(filenames)):
                cp_dict[fn]=frame2second_proposal(FDN_cp_list[idx], rso=rso)
                seg_score_dict[fn]=seg_score_np[idx][:,1].ravel()
                seg_tar_dict[fn]=seg_target_list[idx].data.cpu().numpy().ravel()
            if all(arr.size == 0 for arr in FDN_cp_list):
                # if the coarse proposal list is null, continue
                continue
            # inference PRN
            ver_pred, reg_pred, _, PRN_cp_list = RPN_model(last_hidden_states, FDN_cp_list, rso)
            # VerH decoder is applied to get the VerH proposal list
            ver_label_pred= (ver_pred > 0.5).int().cpu().numpy()
            cp_ver_list=decoder_ver(PRN_cp_list, ver_label_pred)
            # save dict for VerH proposal
            for idx,fn in enumerate(np.array(filenames)):
                ver_dict[fn]=frame2second_proposal(cp_ver_list[idx],rso=rso)
            # get the positive proposals
            sampled_pos_inds = np.where(ver_label_pred == 1)[0] 
            PRN_cp_flatten = np.concatenate([arr for arr in PRN_cp_list if arr.size > 0]).reshape(-1, 3)
            # Compute the proposal lengths
            # Forgery segments smaller than two frames will not be used in the regression process
            lengths = PRN_cp_flatten[sampled_pos_inds][:, 2] - PRN_cp_flatten[sampled_pos_inds][:, 1]
            short_inds = np.where(lengths < 2)[0]
            fp_pred_list,fp_short_list=[np.array([]),np.array([])],[np.array([]),np.array([])]
            if short_inds.size>0: # only verification decoder applied
                sampled_pos_short_inds=sampled_pos_inds[short_inds]
                sampled_pos_inds = np.delete(sampled_pos_inds, short_inds) # remove the short inds
                short_label_pred = np.zeros(len(ver_label_pred)) 
                short_label_pred [sampled_pos_short_inds] = 1
                fp_short_list=decoder_ver(PRN_cp_list, short_label_pred)
            if sampled_pos_inds.size >0: # use regression and verification decoders
                pos_label_pred = np.zeros(len(ver_label_pred))
                pos_label_pred [sampled_pos_inds] = 1
                fp_pred_flatten = decoder_reg(PRN_cp_flatten, reg_pred.data.cpu().numpy())
                fp_pred_flatten[fp_pred_flatten[:, 2] == 0] = PRN_cp_flatten[fp_pred_flatten[:, 2] == 0]
                fp_pred_list= decoder_ver2(fp_pred_flatten, pos_label_pred, FDN_cp_list)
            # save dict for fine-grained (VerH+RegH) proposals
            for fn, fp_short, fp_pred in zip(filenames, fp_short_list, fp_pred_list):
                if fp_short.size == 0:
                    combined = fp_pred
                elif fp_pred.size == 0:
                    combined = fp_short
                else:
                    combined = vstack_twolist(fp_short, fp_pred)
                regver_dict[fn] = frame2second_proposal(combined, rso=rso)
    print("1. Coarse-grained Proposals")
    mAP_CP=eval_TFL(test_gt_dict,cp_dict)
    print("2. Coarse-grained Proposals + VerH")
    mAP_CPVH=eval_TFL(test_gt_dict,ver_dict)
    print("3. Fine-grained Proposals")
    mAP_FP=eval_TFL(test_gt_dict,regver_dict)
    return mAP_FP
