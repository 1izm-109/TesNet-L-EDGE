import os, sys, pdb
import argparse
from data import make_data_loader,get_transform,COCO2014,collate_fn
import numpy as np
from torch.utils.data import DataLoader
import warnings
from trainer import Trainer
import torch
import torch.backends.cudnn as cudnn
import random
import losses
from GCNResNet import tresnet_with_outputlayer
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./datasets/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--epoch_step', default=[], type=int, nargs='+', help='number of epochs to change learning rate')
# parser.add_argument('--device_ids', default=[0], type=int, nargs='+', help='number of epochs to change learning rate')
parser.add_argument('-b', '--batch-size', default=56, type=int)
parser.add_argument('-j', '--num_workers', default=6, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=100, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)#0.05
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

''' Train setting '''
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014')
parser.add_argument('--model_name', type=str, default='Our method')
parser.add_argument('--save_dir', default='./work/coco2014/', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 64)')

def dict_add(dict1,dict2):
    for k,v in dict2.items():
        if 'dict' in type(dict1[k]).__name__ or 'Dict' in type(dict1[k]).__name__:
            dict_add(dict1[k],dict2[k])
        else:
            dict1[k] += v
def dict_div(dict1,d):
    for k,v in dict1.items():
        if 'dict' in type(dict1[k]).__name__ or 'Dict' in type(dict1[k]).__name__:
            dict_add(dict1[k],d)
        else:
            dict1[k] = v/d  

            
def main(args):
    args.seed=20

    args.data_root_dir='./data'
    args.data='COCO2014'
    args.resume = './work/coco2014/checkpoint_best9005.pth'

    args.evaluate =True
    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    is_train = True if not args.evaluate else False
    
    transform = get_transform(args, is_train=True)
    train_dataset= COCO2014('./data/COCO/coco_train2014/data', phase='train', transform=transform)#训练集位置
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=True)
    transform = get_transform(args, is_train=False)
    val_dataset=COCO2014('./data/COCO/coco_val2014/data', phase='val', transform=transform)#验证集位置
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=False)
    num_classes=train_dataset.num_classes 
    
    model = tresnet_with_outputlayer(model='tresnet-l', num_classes=num_classes, t=0.4,
                                        adj_file='data/coco/coco_adj.pkl', pretrained=True,
                                        output_layer=True,hidden_num=768)

    criterion = losses.AsymmetricLoss(
         gamma_neg=4, gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
     )

    trainer = Trainer(model, criterion, train_loader, val_loader, args)
    if is_train:
        trainer.train()
    else:
        trainer.validate()



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    