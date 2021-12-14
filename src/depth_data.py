import torch
import os
import numpy as np
from PIL import Image
import cv2

from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, get_nusc_maps, get_local_map


class SimData(torch.utils.data.Dataset):
    def __init__(self, folder, is_train, data_aug_conf, grid_conf):
        self.folder = folder
        
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        
        
        self.ixes = self.prepro()
        self.length = len(self.ixes)

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

    def prepro(self):

        return [{'rgb': {'L': os.path.join(self.folder,'RGB','L',str(i),'.jpeg'), \
            'R' : os.path.join(self.folder,'RGB','R',str(i),'.jpeg') }, \
            'depth': {'L': os.path.join(self.folder,'DEPTH','L',str(i),'.jpeg'), \
            'R': os.path.join(self.folder,'DEPTH','R',str(i),'.jpeg') } \
            } for i in range(160)]


    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        for cam in cams:
            imgname = rec['rgb'][cam]
            img = Image.open(imgname)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, _, _ = img_transform(img, resize=resize,resize_dims=resize_dims,crop=crop,flip=flip,rotate=rotate)
            imgs.append(normalize_img(img))

        return torch.stack(imgs)

    def get_depthmap(self, rec, cams):
        depths = []
        for cam in cams:
            depthname = rec['depth'][cam]
            depth = Image.open(depthname)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            depth, _, _ = img_transform(depth, resize=resize,resize_dims=resize_dims,crop=crop,flip=flip,rotate=rotate)
            depths.append(normalize_img(depth))

        return torch.stack(depths)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""SimData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)

                                                     
            


class DepthData(SimData): #torch.utils.data.Dataset
    def __init__(self, *args, **kwargs):
        super(DepthData, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        rec = self.ixes[index]
        cams = self.choose_cams()
        imgs = self.get_image_data(rec, cams)
        depthmap = self.get_depthmap(rec, cams)
        
        return imgs, depthmap



def compile_depth_data(dataroot, data_aug_conf, grid_conf, bsz, nworkers):
    folder = dataroot
    traindata = DepthData(folder, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = DepthData(folder, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    return trainloader, valloader