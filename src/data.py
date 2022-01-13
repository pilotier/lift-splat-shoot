"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, get_nusc_maps, get_local_map



class SimData(torch.utils.data.Dataset):
    def __init__(self, folder, is_train, data_aug_conf, grid_conf, nusc_maps=None):
        self.folder = folder
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.cam_index = {'CAM_FRONT_LEFT':6, 'CAM_FRONT':1, 'CAM_FRONT_RIGHT':2,
            'CAM_BACK_LEFT':5, 'CAM_BACK':4, 'CAM_BACK_RIGHT':3}

        # cam_index = {
        #     6 : 'CAM_FRONT_LEFT',
        #     1 : 'CAM_FRONT',
        #     2 : 'CAM_FRONT_RIGHT',
        #     5 : 'CAM_BACK_LEFT',
        #     4 : 'CAM_BACK',
        #     3 : 'CAM_BACK_RIGHT'    
        # }
        
        self.ixes = self.prepro()

        self.length = len(self.ixes)

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.cam_id = {'CAM_FRONT_LEFT':5, 'CAM_FRONT':0, 'CAM_FRONT_RIGHT':1,
            'CAM_BACK_LEFT':4, 'CAM_BACK':3, 'CAM_BACK_RIGHT':2}
        
        
        self.rots = torch.tensor([[[ 0.8225,  0.0065,  0.5687],
                        [-0.5687,  0.0164,  0.8224],
                        [-0.0040, -0.9998,  0.0172]],

                        [[ 0.0103,  0.0084,  0.9999],
                        [-0.9999,  0.0123,  0.0102],
                        [-0.0122, -0.9999,  0.0086]],

                        [[-0.8440,  0.0165,  0.5361],
                        [-0.5361,  0.0036, -0.8441],
                        [-0.0158, -0.9999,  0.0058]],

                        [[ 0.9479, -0.0089, -0.3185],
                        [ 0.3186,  0.0188,  0.9477],
                        [-0.0025, -0.9998,  0.0207]],

                        [[ 0.0092, -0.0068, -0.9999],
                        [ 0.9999,  0.0113,  0.0091],
                        [ 0.0112, -0.9999,  0.0069]],

                        [[-0.9237, -0.0026, -0.3830],
                        [ 0.3830, -0.0114, -0.9237],
                        [-0.0020, -0.9999,  0.0116]]])

        self.trans = torch.tensor([[ 1.5753,  0.5005,  1.5070],
                        [ 1.7220,  0.0048,  1.4949],
                        [ 1.5808, -0.4991,  1.5175],
                        [ 1.0485,  0.4831,  1.5621],
                        [ 0.0552,  0.0108,  1.5679],
                        [ 1.0595, -0.4672,  1.5505]])
        
        self.intrins = torch.tensor([[[1.2579e+03, 0.0000e+00, 8.2724e+02],
                        [0.0000e+00, 1.2579e+03, 4.5092e+02],
                        [0.0000e+00, 0.0000e+00, 1.0000e+00]],

                        [[1.2528e+03, 0.0000e+00, 8.2659e+02],
                        [0.0000e+00, 1.2528e+03, 4.6998e+02],
                        [0.0000e+00, 0.0000e+00, 1.0000e+00]],

                        [[1.2567e+03, 0.0000e+00, 8.1779e+02],
                        [0.0000e+00, 1.2567e+03, 4.5195e+02],
                        [0.0000e+00, 0.0000e+00, 1.0000e+00]],

                        [[1.2550e+03, 0.0000e+00, 8.2958e+02],
                        [0.0000e+00, 1.2550e+03, 4.6717e+02],
                        [0.0000e+00, 0.0000e+00, 1.0000e+00]],

                        [[7.9689e+02, 0.0000e+00, 8.5778e+02],
                        [0.0000e+00, 7.9689e+02, 4.7688e+02],
                        [0.0000e+00, 0.0000e+00, 1.0000e+00]],

                        [[1.2500e+03, 0.0000e+00, 8.2538e+02],
                        [0.0000e+00, 1.2500e+03, 4.6255e+02],
                        [0.0000e+00, 0.0000e+00, 1.0000e+00]]])
        
        self.post_rots = [[[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]],

                            [[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]],

                            [[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]],

                            [[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]],

                            [[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]],

                            [[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]]]
        
        self.post_trans = [[0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.]]
    

        print(self)

    def prepro(self):
        # print("In prepro")
        cameras = {}
        for key,cam in self.cam_index.items():
            path = self.folder + '/' + str(cam) + "/*.jpeg"
            files = glob(path)
            cameras[cam] = files
        bev_files = glob(self.folder + "/BEV/*.png")
        cameras['BEV'] = bev_files
        ls = []
        length = len(cameras['BEV'])

        for i in range(length):
            dct = {}
            dct['Frame'] = i
            data = {}
            for key,cam in self.cam_index.items():
                frame = "CAM"+str(cam)+"_000" + "%03d" % i
                data[key] = [i for i in cameras[cam] if frame in i][0]
            bev_frame = "BEV_000" + "%03d" % i
            data['BEV'] = [i for i in cameras['BEV'] if bev_frame in i][0]
            dct['data'] = data
            ls.append(dct)

        return ls

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
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        # print(rec)
        for cam in cams:
            
            imgname = rec['data'][cam]
            img = Image.open(imgname)#.convert('RGB')
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                    rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(self.intrins[self.cam_id[cam]])
            rots.append(self.rots[self.cam_id[cam]])
            trans.append(self.trans[self.cam_id[cam]])
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    
    def get_binmap(self, rec):
        
        imgname = rec['data']['BEV']
        # print(imgname)
        img = Image.open(imgname)
        img = img.rotate(180)

        
        newsize = (self.nx[0], self.nx[1])
        img = img.resize(newsize)
        img = np.array(img)

        #vehicle label
        img_vehicle = np.zeros((self.nx[0], self.nx[1]))
        img_vehicle = (img[:,:,0] < 200) * (img[:,:,1] > 200) * (img[:,:,2] < 200) 
        
        #road_segment
        img_road_segment = np.zeros((self.nx[0], self.nx[1]))
        img_road_segment = (img[:,:,0] > 200)

        #pedestrians
        img_lane_divider = np.zeros((self.nx[0], self.nx[1]))
        img_road_segment = (img[:,:,2] > 200)

        return torch.Tensor(np.stack([img_vehicle,img_road_segment,img_lane_divider]))

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


class SimSegmentationData(SimData): #torch.utils.data.Dataset
    def __init__(self, *args, **kwargs):
        super(SimSegmentationData, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        rec = self.ixes[index]
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        #binimg = self.get_binimg(rec)
        binmap = self.get_binmap(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binmap


def compile_sim_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, map_folder='', length=1):
    folder = dataroot

    maps = None
    if map_folder:
        maps = map_folder
    
    traindata = SimSegmentationData(folder, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = SimSegmentationData(folder, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)
    # print(trainloader,valloader)

    return trainloader, valloader


def worker_rnd_init(x):
    np.random.seed(13 + x)
