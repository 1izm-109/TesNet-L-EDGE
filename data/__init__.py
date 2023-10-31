import os, sys, pdb
from PIL import Image
from PIL import ImageDraw
import random

import torch
from torch.utils.data import DataLoader
from randaugment import RandAugment
from copy import deepcopy
import torchvision.transforms as transforms
import numpy as np

from .coco import COCO2014
from .voc import VOC2007, VOC2012

data_dict = {'COCO2014': COCO2014,
            'VOC2007': VOC2007,
            'VOC2012': VOC2012}

def collate_fn(batch):
    ret_batch = dict()
    for k in batch[0].keys():
        if k == 'image' or k == 'target':
            ret_batch[k] = torch.cat([b[k].unsqueeze(0) for b in batch])
        else:
            ret_batch[k] = [b[k] for b in batch]
    return ret_batch

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__


def get_transform(args, is_train=True):
    if is_train:
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(args.image_size, scale=(0.1, 1.5), ratio=(1.0, 1.0)),
            # transforms.RandomResizedCrop(args.image_size, scale=(0.1, 2.0), ratio=(1.0, 1.0)),
            transforms.Resize((args.image_size, args.image_size)),
            # transforms.Resize((args.image_size, args.image_size)),
            # RandAugment(),
            # SLCutoutPIL(1,-1),
            CutoutPIL(0.5),# 0.2 0.1 均为 94.7
            RandAugment(),
            # CutoutPIL(0.5),
            # SLCutoutPIL(n_holes=1, length=-1),
            # MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            # transforms.RandomHorizontalFlip(),
            # transforms.AutoAugment(),
            # # SLCutoutPIL(args.n_holes, args.length),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.3,
                                      # contrast=0.3,
                                      # saturation=0.3,
                                      # hue=0),
            
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Cutout(n_holes=1, length=64)
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size,args.image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def make_data_loader(args, is_train=True):
    root_dir = os.path.join(args.data_root_dir, args.data)

    # Build val_loader
    transform = get_transform(args, is_train=False)
    if args.data == 'COCO2014':
        val_dataset = COCO2014(root_dir, phase='val', transform=transform)
    elif args.data in ('VOC2007', 'VOC2012'):
        val_dataset = data_dict[args.data](root_dir, phase='test', transform=transform)
    else:
        raise NotImplementedError('Value error: No matched dataset!')
    
    num_classes = val_dataset[0]['target'].size(-1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=False)
    
    if not is_train:
        return None, val_loader, num_classes
    
    # Build train_loader
    transform = get_transform(args, is_train=True)
    if args.data == 'COCO2014':
        train_dataset = COCO2014(root_dir, phase='train', transform=transform)
    elif args.data in ('VOC2007', 'VOC2012'):
        train_dataset = data_dict[args.data](root_dir, phase='trainval', transform=transform)
    else:
        raise NotImplementedError('Value error: No matched dataset!')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, 
                            collate_fn=collate_fn, drop_last=True)
    

    return train_loader, val_loader, num_classes


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

class SLCutoutPIL(object):
    def __init__(self, n_holes, length, cut_fact=None):
        self.n_holes = n_holes
        self.length = length
        self.cut_fact = cut_fact
        if self.cut_fact is not None:
            assert length < 0, "length must be set to -1 but {} if cut_fact is not None!".format(length)

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        if self.cut_fact is not None:
            h_cutout = int(self.cutout_factor * h)
            w_cutout = int(self.cutout_factor * w)
        else:
            h_cutout = int(self.length)
            w_cutout = int(self.length)
        for i in range(self.n_holes):
            y_c = np.random.randint(h)
            x_c = np.random.randint(w)

            y1 = np.clip(y_c - h_cutout // 2, 0, h)
            y2 = np.clip(y_c + h_cutout // 2, 0, h)
            x1 = np.clip(x_c - w_cutout // 2, 0, w)
            x2 = np.clip(x_c + w_cutout // 2, 0, w)
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

class SLCutoutPIL(object):
    def __init__(self, n_holes, length, cut_fact=None):
        self.n_holes = n_holes
        self.length = length
        self.cut_fact = cut_fact
        if self.cut_fact is not None:
            assert length < 0, "length must be set to -1 but {} if cut_fact is not None!".format(length)

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        if self.cut_fact is not None:
            h_cutout = int(self.cutout_factor * h)
            w_cutout = int(self.cutout_factor * w)
        else:
            h_cutout = int(self.length)
            w_cutout = int(self.length)
        for i in range(self.n_holes):
            y_c = np.random.randint(h)
            x_c = np.random.randint(w)

            y1 = np.clip(y_c - h_cutout // 2, 0, h)
            y2 = np.clip(y_c + h_cutout // 2, 0, h)
            x1 = np.clip(x_c - w_cutout // 2, 0, w)
            x2 = np.clip(x_c + w_cutout // 2, 0, w)
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

    
class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
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