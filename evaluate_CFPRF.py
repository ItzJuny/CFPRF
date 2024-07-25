import torch,argparse
import numpy as np
from libs.dataloader.data_io import get_dataloader
from models.FDN import * 
from models.PRN import * 
from libs import tool
from libs.startup_config import set_random_seed
import glob
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"
############EVAL CFPRF################

    
def Inference_CFPRF(eval_dlr, FDN_model, RPN_model,  rso,  device):
    print("++++++++++++++++++inference++++++++++++++++++")
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
                # If the coarse proposal list is null, continue
                continue
            # inference PRN
            ver_pred, reg_pred, _, PRN_cp_list = RPN_model(last_hidden_states, FDN_cp_list, rso)
            # VerH Decoder is applied to get the VerH proposal list
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
    return seg_score_dict, seg_tar_dict, cp_dict, ver_dict, regver_dict

def FDN_performance_excel(seg_score_dict,seg_tar_dict):
    Seer_value,Sacc,f1,pre,rec, Sauc=eval_PFD(seg_score_dict,seg_tar_dict)
    savecontent=pd.DataFrame()
    savecontent['Seer']=["%.2f"%float(Seer_value)]
    savecontent['Sauc']=["%.2f"%float(Sauc)]
    savecontent['P']=["%.2f"%float(pre)]
    savecontent['R']=["%.2f"%float(rec)]
    savecontent['F1']=["%.2f"%float(f1)]
    savecontent['Sacc']=["%.2f"%float(Sacc)]
    return savecontent

def PRN_performance_excel(gt_dict,proposal_dict):
    mAP_value, ap_score, ar_score=eval_TFL(gt_dict, proposal_dict, detail=True)
    savecontent=pd.DataFrame()
    savecontent['mAP']=["%.2f"%float(mAP_value*100)]
    for ap in ap_score:
        savecontent['AP@'+str(ap)]=["%.2f"%float(ap_score[ap]*100)]
    for ar in sorted(ar_score):
        savecontent['AR@'+str(ar)]=["%.2f"%float(ar_score[ar]*100)]
    return savecontent

if __name__ == '__main__':
    parser = argparse.ArgumentParser('python evaluate_CFPRF.py --dn HAD/PS/LAVDF')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default="./result/") # ['HAD','PS','LAVDF']
    parser.add_argument('--dn', type=str, default="HAD") # ['HAD','PS','LAVDF']
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--rso', type=int, default=20)
    parser.add_argument('--glayer', type=int, default=1) 
    parser.add_argument('--eval', action='store_true', default=True) 
    args = parser.parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """loading dataset"""
    test_gt_dict, test_dlr=get_dataloader(batch_size=1,part="test",dn=args.dn,rso=args.rso)
    """loading FDN model"""
    FDN_model = CFPRF_FDN(seq_len=args.seql, gmlp_layers=args.glayer).to(device)
    """ 
    ./checkpoints
    ├── 1FDN_HAD.pth
    ├── 1FDN_LAVDF.pth
    ├── 1FDN_PS.pth
    ├── 2PRN_HAD.pth
    ├── 2PRN_LAVDF.pth
    ├── 2PRN_PS.pth
    """
    FDN_checkpoint="./checkpoints/1FDN_%s.pth"%(args.dn)
    FDN_model.load_state_dict(torch.load(FDN_checkpoint))
    """loading PRN model"""
    RPN_model = CFPRF_PRN(device=device).to(device)
    PRN_modelpath="./checkpoints/2PRN_%s.pth"%(args.dn)
    RPN_model.load_state_dict(torch.load(PRN_modelpath))
    """makedir"""
    dict_save_path=os.path.join(args.save_path,'dict/%s_'%(args.dn))
    csv_save_path=os.path.join(args.save_path,'pd/%s_'%(args.dn))
    os.makedirs(os.path.dirname(dict_save_path),exist_ok=True)
    os.makedirs(os.path.dirname(csv_save_path),exist_ok=True)
    ###########INFERENCE#############
    if args.eval:
        seg_score_dict, seg_tar_dict, cp_dict, ver_dict, regver_dict=Inference_CFPRF(test_dlr, FDN_model, RPN_model,  args.rso, device)
        writenpy(dict_save_path+'seg_tar_dict.npy',seg_tar_dict)
        writenpy(dict_save_path+'seg_score_dict.npy',seg_score_dict)
        writenpy(dict_save_path+'cp_dict.npy',cp_dict)
        writenpy(dict_save_path+'ver_dict.npy',ver_dict)
        writenpy(dict_save_path+'regver_dict.npy',regver_dict)
    else:
        seg_score_dict=readnpy(dict_save_path+'seg_score_dict.npy',seg_score_dict)
        seg_tar_dict=readnpy(dict_save_path+'seg_tar_dict.npy',seg_tar_dict)
        cp_dict=readnpy(dict_save_path+'cp_dict.npy',cp_dict)
        ver_dict=readnpy(dict_save_path+'ver_dict.npy',ver_dict)
        regver_dict=readnpy(dict_save_path+'regver_dict.npy',regver_dict)
    ###########PFD#############
    savecontent0=FDN_performance_excel(seg_score_dict,seg_tar_dict)
    savecontent0.to_csv(csv_save_path+'PFD_results.csv')
    print("PFD",savecontent0)
    ###########TFL_CP#############
    savecontent1=PRN_performance_excel(test_gt_dict,cp_dict)
    savecontent1.to_csv(csv_save_path+'TFL_CP_results.csv')
    print("CP",savecontent1)
    ###########TFL_VER#############
    savecontent2=PRN_performance_excel(test_gt_dict,ver_dict)
    savecontent2.to_csv(csv_save_path+'TFL_VER_results.csv')
    print("VER",savecontent2)
    ###########TFL_RP#############
    savecontent3=PRN_performance_excel(test_gt_dict,regver_dict)
    savecontent3.to_csv(csv_save_path+'TFL_RP_results.csv')
    print("RP",savecontent3)
