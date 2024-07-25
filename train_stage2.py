import torch
import argparse
from models.FDN import * 
from models.PRN import * 
from libs import tool
from libs.dataloader.data_io import get_dataloader
from libs.startup_config import set_random_seed
from libs.wrapper_PRN import *
import warnings
warnings.filterwarnings("ignore")
__author__ = "Junyan Wu"
__email__ = "wujy298@mail2.sysu.edu.cn"
############TRAIN PRN################
if __name__ == '__main__':
    parser = argparse.ArgumentParser('python main_stage2.py --dn PS')
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--wd', type=float, default=0.001) 
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--warm_epoch', type=int, default=5)
    parser.add_argument('--stop_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dn', type=str, default="PS")
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--rso', type=int, default=20)
    parser.add_argument('--glayer', type=int, default=1) 
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    assert args.dn in ['PS', 'HAD','LAVDF']
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """loading dataset"""
    _, train_dlr=get_dataloader(batch_size=args.bs,part="train",dn=args.dn,rso=args.rso)
    dev_gt_dict, dev_dlr=get_dataloader(batch_size=1,part="dev",dn=args.dn,rso=args.rso)
    """loading FDN_model"""
    print("loading FDN_model")
    FDN_model = CFPRF_FDN(seq_len=args.seql, gmlp_layers=args.glayer).to(device)
    FDN_modelpath = "./checkpoints/1FDN_%s.pth"%(args.dn)
    FDN_model.load_state_dict(torch.load(FDN_modelpath))
    for name, param in FDN_model.named_parameters(): # freeze
        param.requires_grad = False
    """loading PRN model"""
    print("loading PRN_model")
    RPN_model = CFPRF_PRN(device=device).to(device)
    """saving model"""
    model_tag = '{}_seed{}_lr{:7f}_wd{}_bs{}_rso{}'.format(os.path.basename(FDN_modelpath).rstrip('.pth'),args.seed, args.lr, args.wd, args.bs, args.rso)
    modelpath="./checkpoints/%s/PRN/%s/"%(args.dn,model_tag)
    os.makedirs(modelpath, exist_ok=True)
    print(modelpath)
    """Training"""
    loss_ver = nn.BCELoss()
    optimizer = torch.optim.Adam(RPN_model.parameters(), lr=args.lr,weight_decay=args.wd)
    best_dev_mAP=0
    stop=0
    for epoch in range(1, args.num_epoch+1):
        if stop>=args.stop_epoch:
            print('Early Stop.')
            sys.exit(0)
        print("-----------------------train epoch= %d -------------------------"%epoch)
        train_loss,_=train_one_epoch(train_dlr, FDN_model, RPN_model, loss_ver, args.rso, device, optimizer)
        if epoch<=args.warm_epoch:
            print('train_loss{:.4f}'.format(train_loss))
            continue
        print("-----------------------dev epoch= %d -------------------------"%epoch)
        dev_fp_mAP=test_one_epoch(dev_dlr, dev_gt_dict, FDN_model, RPN_model, args.rso, device)
        print('train_loss{:.4f}\tdev_coarse_mAP{:.4f}'.format(train_loss,dev_fp_mAP))
        if dev_fp_mAP >= best_dev_mAP:
            best_dev_mAP, stop=dev_fp_mAP, 0
            if args.save:
                torch.save(RPN_model.state_dict(), os.path.join(modelpath, 'e{}_FPmAP{:.3f}.pth'.format(epoch,dev_fp_mAP)))
        else:
            stop+=1
            continue