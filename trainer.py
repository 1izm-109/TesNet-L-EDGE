import os, sys, pdb
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torchnet as tnt
# import torchvision.transforms as transforms
from torch.autograd import Variable
# from torch.optim import lr_scheduler
from util import AverageMeter, AveragePrecisionMeter
from datetime import datetime
# from pprint import pprint
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F
from data import ModelEma
from coco_util import *
# from GCNResNet import *
class Trainer(object):
    def __init__(self, model, criterion, train_loader, val_loader, args):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        # pprint (self.args)
        print('--------Args Items----------')
        for k, v in vars(self.args).items():
            print('{}: {}'.format(k, v))
        print('--------Args Items----------\n')

    def initialize_optimizer_and_scheduler(self):
        def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
            
        # parameters = add_weight_decay(self.model, self.args.weight_decay)
        # self.optimizer = torch.optim.Adam(params=parameters, lr=self.args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
        parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]},
        ]
        # self.optimizer = torch.optim.AdamW(parameters, self.args.lr, weight_decay=0)
        self.optimizer = torch.optim.AdamW(parameters, self.args.lr, weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.get_config_optim(self.args.lr, self.args.lrp), 
                                 # lr=self.args.lr, 
                                 # momentum=self.args.momentum, 
                                 # weight_decay=self.args.weight_decay)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs, pct_start=0.2)

    def initialize_meters(self):
        self.meters = {}
        # meters
        self.meters['loss'] = AverageMeter('loss')
        self.meters['ap_meter'] = AveragePrecisionMeter()
        # time measure
        self.meters['batch_time'] = AverageMeter('batch_time')
        self.meters['data_time'] = AverageMeter('data_time')

    def initialization(self, is_train=False):
        """ initialize self.model and self.criterion here """
        
        if is_train:
            self.start_epoch = 0
            self.epoch = 0
            self.end_epoch = self.args.epochs
            self.best_score = 0.
            self.lr_now = self.args.lr

            # initialize some settings
            self.initialize_optimizer_and_scheduler()

        self.initialize_meters()

        # load checkpoint if args.resume is a valid checkpint file.
        if os.path.isfile(self.args.resume) and self.args.resume.endswith('pth'):
            self.load_checkpoint()
        
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = torch.nn.DataParallel(self.model).cuda()
            # self.model.load_state_dict(torch.load('./work/voc2007/checkpoint_best9706.pth')['state_dict'])
            # self.model.load_state_dict(torch.load('./work/coco2014/checkpoint_best9008.pth')['state_dict'],True)
            # self.model.load_state_dict(torch.load('./work/voc2012/checkpoint_best-Copy1.pth')['state_dict'],True)
            self.criterion = self.criterion.cuda()
            self.ema = ModelEma(self.model, 0.9997)
        self.scaler = torch.cuda.amp.GradScaler();

    def reset_meters(self):
        for k, v in self.meters.items():
            self.meters[k].reset()

    def on_start_epoch(self):
        self.reset_meters()

    def on_end_epoch(self, is_train=False):

        if is_train:
            # maybe you can do something like 'print the training results' here.
            return 
        else:
            # map = self.meters['ap_meter'].value().mean()
            ap =  self.meters['ap_meter'].value()
            print (ap)
            map = ap.mean()
            loss = self.meters['loss'].average()
            data_time = self.meters['data_time'].average()
            batch_time = self.meters['batch_time'].average()

            # OP, OR, OF1, CP, CR, CF1 = self.meters['ap_meter'].overall()
            OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.meters['ap_meter'].overall_topk(3)

            print('* Test\nLoss: {loss:.4f}\t mAP: {map:.4f}\t' 
                    'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
                    loss=loss, map=map, data_time=data_time, batch_time=batch_time))
#             print('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
#                     'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
#                     OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            print('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
                    'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
                    OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
                    
            return map

    def on_forward(self, inputs, targets, is_train, ema_model=False):
        inputs = Variable(inputs).float()
        targets = Variable(targets).float()
        torch.cuda.empty_cache()
        if not is_train:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if ema_model:
                        outputs = self.ema.module(inputs)
                    else:
                        outputs = self.model(inputs)
                # loss = self.criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                # loss = self.criterion(outputs, targets)
        loss = self.criterion(outputs, targets)

        self.meters['loss'].update(loss.item(), inputs.size(0))

        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_clip_grad_norm)
            # self.optimizer.step()
            self.ema.update(self.model)
            self.scheduler.step()
            
        return outputs
    
    def adjust_learning_rate(self):
        """ Sets learning rate if it is needed """
        lr_list = []
        decay = 0.1 if sum(self.epoch == np.array(self.args.epoch_step)) > 0 else 1.0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])

        return np.unique(lr_list)

    def train(self):
        self.initialization(is_train=True)

        for epoch in range(self.start_epoch, self.end_epoch):
            if epoch == 25:
                break
            self.lr_now = self.adjust_learning_rate()
            print ('Lr: {}'.format(self.lr_now))
            
            # print(self.model.module.output_layer.embed.weight.detach().cpu().numpy())
            self.epoch = epoch
            # train for one epoch
            self.run_iteration(self.train_loader, is_train=True)
            
            if epoch+1 == 5: # voc2007 voc2012 都是5 coco2014不需要 注释掉
                self.ema = ModelEma(self.model, 0.9997)
            
            # evaluate on validation set
            score = self.run_iteration(self.val_loader, is_train=False)
            ema_score = self.run_iteration(self.val_loader, is_train=False, ema_model=True)
            state_dict = self.ema.module.state_dict()
            if score > ema_score:
                score = score
                state_dict = self.model.module.state_dict()
            else:
                score = ema_score
                state_dict = self.ema.module.state_dict()
            #record best score, save checkpoint and result
            is_best = score > self.best_score
            self.best_score = max(score, self.best_score)
#             checkpoint = {
#                 'epoch': epoch + 1, 
#                 'model_name': self.args.model_name,
#                 'state_dict': state_dict,
#                 'best_score': 0
# #                 'best_score': self.best_score
#                 }
            checkpoint = {
                'epoch': epoch + 1, 
                'model_name': self.args.model_name,
                'state_dict': state_dict,
                'best_score': score
                # 'best_score': self.best_score
                }
            model_dir = self.args.save_dir
            # assert os.path.exists(model_dir) == True
            self.save_checkpoint(checkpoint, model_dir, is_best)
            self.save_result(model_dir, is_best)
            # self.save_checkpoint(checkpoint, model_dir, True)
            # self.save_result(model_dir, True)

            print(' * best mAP={best:.4f}'.format(best=self.best_score))

        return self.best_score

    def run_iteration(self, data_loader, is_train=True, ema_model=False):
        self.on_start_epoch()
        
        if not is_train:
            prec = COCOAverageMeter()
            rec = COCOAverageMeter()
            Sig = torch.nn.Sigmoid()
            tp, fp, fn, tn, count = 0, 0, 0, 0, 0
            preds = []
            target_array = []
    
        if not is_train:
            data_loader = tqdm(data_loader, desc='Validate')
            self.model.eval()
        else:
            self.model.train()

        st_time = time.time()
        for i, data in enumerate(data_loader):

            # measure data loading time
            data_time = time.time() - st_time
            self.meters['data_time'].update(data_time)

            # inputs, targets, targets_gt, filenames = self.on_start_batch(data)
            inputs = data['image']
            targets = data['target']

            # for voc
            labels = targets.clone()
            targets[targets==0] = 1
            targets[targets==-1] = 0
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = self.on_forward(inputs, targets, is_train=is_train, ema_model=ema_model)
            # measure elapsed time
            if not is_train:
                outputs = Sig(outputs)
                outputs= outputs.cpu()
                targets = targets.cpu()
                # print(output)
                # print(target)
                # for mAP calculation
                preds.append(outputs.cpu())

                target_array.append(targets.cpu())
                # print(target)
                # n, c = output.size()
                # scores = torch.zeros((n, c))
                # index = output.data.topk(3, 1, True, True)[1]
                # tmp = output
                #scores[tmp>=0.75] = 1
                # for j in range(output.shape[0]):
                    # for ind in index[j]:
                        # scores[j, ind] = 1 if tmp[j, ind] >= 0.75 else 0

                # measure accuracy and record loss
                pred = outputs.data.gt(0.75).long()
                # pred = scores.long()
                tp += (pred + targets).eq(2).sum(dim=0)
                fp += (pred - targets).eq(1).sum(dim=0)
                fn += (pred - targets).eq(-1).sum(dim=0)
                tn += (pred + targets).eq(0).sum(dim=0)
                count += inputs.size(0)

                this_tp = (pred + targets).eq(2).sum()
                this_fp = (pred - targets).eq(1).sum()
                this_fn = (pred - targets).eq(-1).sum()
                this_tn = (pred + targets).eq(0).sum()

                this_prec = this_tp.float() / (
                    this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
                this_rec = this_tp.float() / (
                    this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

                prec.update(float(this_prec), inputs.size(0))
                rec.update(float(this_rec), inputs.size(0))

                p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                                     i] > 0 else 0.0
                       for i in range(len(tp))]
                r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                                     i] > 0 else 0.0
                       for i in range(len(tp))]
                f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
                       i in range(len(tp))]

                mean_p_c = sum(p_c) / len(p_c)
                mean_r_c = sum(r_c) / len(r_c)
                mean_f_c = sum(f_c) / len(f_c)

                p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
                r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
                f_o = 2 * p_o * r_o / (p_o + r_o)
            
            batch_time = time.time() - st_time
            self.meters['batch_time'].update(batch_time)

            self.meters['ap_meter'].add(outputs.data, labels.data, data['name'])
            st_time = time.time()
            
            if is_train and i % self.args.display_interval == 0:
                print ('{}, {} Epoch, {} Iter, Loss: {:.4f}, Data time: {:.4f}, Batch time: {:.4f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  self.epoch+1, i, 
                        self.meters['loss'].value(), self.meters['data_time'].value(), 
                        self.meters['batch_time'].value()))
        if not is_train:
            print(
            '--------------------------------------------------------------------')
            print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
              .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
        # model_dir = self.args.save_dir
        # self.save_result(model_dir,True)
        return self.on_end_epoch(is_train=is_train)

    def validate(self):
        self.initialization(is_train=False)

        map = self.run_iteration(self.val_loader, is_train=False)

        model_dir = os.path.dirname(self.args.resume)
        assert os.path.exists(model_dir) == True
        self.save_result(model_dir, is_best=False)

        return map

    def load_checkpoint(self):
        print("* Loading checkpoint '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        model_dict = self.model.state_dict()
        model_dict_key = { k:False for _,k in enumerate(model_dict.keys())}
        for k, v in checkpoint['state_dict'].items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                model_dict_key[k] = True
            elif k.replace('module.','') in model_dict and v.shape == model_dict[k.replace('module.','')].shape:
                model_dict[k.replace('module.','')] = v
                model_dict_key[k.replace('module.','')] = True
            else:
                print ('\tMismatched layers: {}'.format(k))
        for a, b in enumerate(model_dict_key):
            if b is False:
                print('checkpoint don`t contain key' + a)
        self.model.load_state_dict(model_dict)

    def save_checkpoint(self, checkpoint, model_dir, is_best=False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # filename = 'Epoch-{}.pth'.format(self.epoch)
        filename = 'checkpoint.pth'
        res_path = os.path.join(model_dir, filename)
        print('Save checkpoint to {}'.format(res_path))
        torch.save(checkpoint, res_path)
        if is_best:
            filename_best = 'checkpoint_best.pth'
            res_path_best = os.path.join(model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)

    def save_result(self, model_dir, is_best=False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # filename = 'results.csv' if not is_best else 'best_results.csv'
        filename = 'results.csv'
        res_path = os.path.join(model_dir, filename)
        print('Save results to {}'.format(res_path))
        with open(res_path, 'w') as fid:
            for i in range(self.meters['ap_meter'].scores.shape[0]):
                fid.write('{},{},{}\n'.format(self.meters['ap_meter'].filenames[i], 
                    ','.join(map(str,self.meters['ap_meter'].scores[i].numpy())), 
                    ','.join(map(str,self.meters['ap_meter'].targets[i].numpy()))))
        
        if is_best:
            filename_best = 'output_best.csv'
            res_path_best = os.path.join(model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)
