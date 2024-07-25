import torch
import argparse
from models.FDN import * 
from libs import tool
from libs.dataloader.data_io import get_dataloader
import libs.loss.p2sgrad as p2sgrad
import libs.loss.crl_loss as crl_loss
from libs.startup_config import set_random_seed
from libs.wrapper_FDN import *
import warnings
warnings.filterwarnings("ignore")
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"
############TRAIN FDN################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('python main_stage1.py --dn PS')
    """For training"""
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--warm_epoch', type=int, default=5)
    parser.add_argument('--stop_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dn', type=str, default="PS")
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--rso', type=int, default=20)
    parser.add_argument('--v1', type=float, default=0.15)
    parser.add_argument('--v2', type=float, default=0.1)
    parser.add_argument('--glayer', type=int, default=1) 
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    assert args.dn in ['PS', 'HAD','LAVDF']
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """loading FDN_model"""
    model = CFPRF_FDN(seq_len=args.seql, gmlp_layers=args.glayer).to(device)
    """loading dataset"""
    _, train_dlr=get_dataloader(batch_size=args.bs,part="train",dn=args.dn,rso=args.rso)
    dev_gt_dict, dev_dlr=get_dataloader(batch_size=1,part="dev",dn=args.dn,rso=args.rso)
    """saving model"""
    model_tag = 'seed{}_lr{:7f}_wd{}_bs{}_Seql{}_Gl{}_Rso{}_v1{}_v2{}'.format(args.seed, args.lr, args.wd, args.bs, args.seql,args.glayer,args.rso,args.v1,args.v2)
    modelpath="./checkpoints/%s/FDN/%s/"%(args.dn,model_tag)
    os.makedirs(modelpath, exist_ok=True)
    print(modelpath)
    """Training"""
    loss_pgMSE=p2sgrad.P2SGradLoss()
    loss_crl=crl_loss.CRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.wd)
    best_dev_eer=999
    best_dev_mAP=0
    stop=0
    for epoch in range(1, args.num_epoch+1):
        if stop>=args.stop_epoch:
            print('Early Stop.')
            sys.exit(0)
        print("-----------------------train epoch= %d -------------------------"%epoch)
        train_loss=train_one_epoch(train_dlr, model, loss_pgMSE, loss_crl, device, optimizer, args.rso, args.v1,args.v2)
        if epoch<=args.warm_epoch:
            print('train_loss{:.4f}'.format(train_loss))
            continue
        print("-----------------------dev epoch= %d -------------------------"%epoch)
        dev_eer,dev_mAP=test_one_epoch(dev_dlr, dev_gt_dict, model, args.rso, device)
        print('train_loss{:.4f}\tdev_eer{:.4f}\tdev_cp_mAP{:.4f}'.format(train_loss,dev_eer,dev_mAP))
        if dev_mAP>=best_dev_mAP and dev_eer<=best_dev_eer+0.1:
            best_dev_eer, best_dev_mAP, stop = dev_eer, dev_mAP, 0
            if args.save:
                torch.save(model.state_dict(), os.path.join(modelpath, 'e{}_devEER{:.3f}_devmAP{:.3f}.pth'.format(epoch,dev_eer,dev_mAP)))
        else:
            stop+=1
            continue
        