import os, sys, pdb
import argparse
from data import make_data_loader,get_transform,collate_fn,VOC2012
from torch.utils.data import DataLoader
import warnings
from trainer import Trainer
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import losses
from GCNResNet import *
parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./datasets/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--epoch_step', default=[], type=int, nargs='+', help='number of epochs to change learning rate')
# parser.add_argument('--device_ids', default=[0], type=int, nargs='+', help='number of epochs to change learning rate')
parser.add_argument('-b', '--batch-size', default=56, type=int)
parser.add_argument('-j', '--num_workers', default=6, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

''' Train setting '''
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014')
parser.add_argument('--model_name', type=str, default='Our method')
parser.add_argument('--save_dir', default='./work/voc2012', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def main(args):
    args.seed=20
    args.data='voc2012'
#     print(torch.load('./work/voc2007/checkpoint_best.pth')['epoch'])
#     args.resume='./work/voc2007/checkpoint_best_val9717.pth'
    # args.evaluate=True
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
    train_dataset= VOC2012('./data/voc2012', phase='train', transform=transform)#训练集位置
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=True)
    transform = get_transform(args, is_train=False)
    val_dataset=VOC2012('./data/voc2012', phase='val', transform=transform)#验证集位置
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=False)
    num_classes = val_dataset[0]['target'].size(-1)

    # model = resnet_with_outputlayer(model='ResNet-101', num_classes=num_classes, t=0.4,
    #                                     adj_file='./work/data/voc/voc_adj.pkl', pretrained=True,
    #                                     output_layer=True,hidden_num=256,bn_freeze=True)
    model = tresnet_with_outputlayer(model='tresnet-l', num_classes=num_classes, t=0.4,
                                        adj_file='data/voc/voc2012_adj.pkl', pretrained=True,
                                        output_layer=True,hidden_num=768,bn_freeze=False)
    # model.load_state_dict(torch.load('./work/voc2007/checkpoint_best9669.pth')['state_dict'], False)
    # criterion = torch.nn.MultiLabelSoftMarginLoss()
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
