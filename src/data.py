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


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, nusc_maps=None):
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.scene2map = {}
        for rec in nusc.scene:
            log = nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']
        
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        if self.nusc.version == 'test':
            return create_splits_scenes()['test']

        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
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
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
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
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)
    
    def get_binmap(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        #img = np.zeros((self.nx[0], self.nx[1]))

        #vehicle label
        img_vehicle = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)
            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img_vehicle, [pts], 1.0)

        # cv2.imwrite('./output/vehicle{}.png'.format(rec['timestamp']),img_vehicle*255)

        #map label
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]
        rot_map = Quaternion(egopose['rotation']).rotation_matrix
        rot_map = np.arctan2(rot_map[1, 0], rot_map[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot_map), np.sin(rot_map)])
        lmap = get_local_map(self.nusc_maps[map_name], center,50.0, ['road_segment','lane'], ['lane_divider','road_divider'])
        #road_segment
        img_road_segment = np.zeros((self.nx[0], self.nx[1]))
        arr_pts=[]
        for pts in lmap['road_segment']:
            pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            arr_pts.append(pts)
        cv2.fillPoly(img_road_segment,arr_pts,1.0)
        #lane
        #lane = np.zeros((self.nx[0], self.nx[1]))
        arr_pts=[]
        for pts in lmap['lane']:
            pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            arr_pts.append(pts)
        # cv2.fillPoly(lane,arr_pts,1.0)
        cv2.fillPoly(img_road_segment,arr_pts,1.0)
        #road_divider
        # img_road_divider = np.zeros((self.nx[0], self.nx[1]))
        # arr_pts=[]
        # for pts in lmap['road_divider']:
        #     pts = np.round(
        #             (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
        #             ).astype(np.int32)
        #     pts[:, [1, 0]] = pts[:, [0, 1]]
        #     arr_pts.append(pts)

        # cv2.polylines(img_road_divider,arr_pts,False,1.0,1)
        #lane_divider

        img_lane_divider = np.zeros((self.nx[0], self.nx[1]))
        arr_pts=[]
        for pts in lmap['lane_divider']:
            pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            arr_pts.append(pts)

        cv2.polylines(img_lane_divider,arr_pts,False,1.0,2)

        # cv2.imwrite('./output/lane_divider{}.png'.format(rec['timestamp']),img_lane_divider*255)
            #plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)

        return torch.Tensor(np.stack([img_vehicle,img_road_segment,img_lane_divider]))

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


class MultiSegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(MultiSegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        #binimg = self.get_binimg(rec)
        binmap = self.get_binmap(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binmap


class SimData(torch.utils.data.Dataset):
    def __init__(self, folder, is_train, data_aug_conf, grid_conf, nusc_maps=None):
        self.folder = folder
        
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        
        self.ixes = self.prepro()
<<<<<<< HEAD
        self.length = len(self.ixes)
=======
>>>>>>> 51cf2c0a1849538467853557f1eb58641ee15710

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.cam_id = {'CAM_FRONT_LEFT':5, 'CAM_FRONT':0, 'CAM_FRONT_RIGHT':1,
            'CAM_BACK_LEFT':4, 'CAM_BACK':3, 'CAM_BACK_RIGHT':2}
        
        self.cam_index = {'CAM_FRONT_LEFT':0, 'CAM_FRONT':1, 'CAM_FRONT_RIGHT':2,
            'CAM_BACK_LEFT':5, 'CAM_BACK':4, 'CAM_BACK_RIGHT':3}
        
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
        for cam in range(1,7):
            # print(cam)
<<<<<<< HEAD
            path = "/home/tanushri/Work/lift-splat-shoot/LSS_TEST2/LSSCAM/" + str(cam) + "/*.jpeg"
            files = glob(path)
            cameras[cam] = files
        bev_files = glob("/home/tanushri/Work/lift-splat-shoot/LSS_TEST2/LSSCAM/BEV/*.jpeg")
        cameras['BEV'] = bev_files

=======
            path = self.folder + '/' + str(cam) + "/*.jpeg"
            files = glob(path)
            cameras[cam] = files
        bev_files = glob(self.folder + "/BEV/*.png")
        cameras['BEV'] = bev_files

        print(cameras['BEV'])

>>>>>>> 51cf2c0a1849538467853557f1eb58641ee15710
        cam_index = {
            6 : 'CAM_FRONT_LEFT',
            1 : 'CAM_FRONT',
            2 : 'CAM_FRONT_RIGHT',
            5 : 'CAM_BACK_LEFT',
            4 : 'CAM_BACK',
            3 : 'CAM_BACK_RIGHT'    
        }

        ls = []
<<<<<<< HEAD
        for i in range(311):
=======
        for i in range(15):
>>>>>>> 51cf2c0a1849538467853557f1eb58641ee15710
            dct = {}
            dct['Frame'] = i
            data = {}
            for cam in cam_index:
                frame = "CAM"+str(cam)+"_000" + "%03d" % i
                data[cam_index[cam]] = [i for i in cameras[cam] if frame in i][0]
            bev_frame = "BEV_000" + "%03d" % i
            data['BEV'] = [i for i in cameras['BEV'] if bev_frame in i][0]
            dct['data'] = data
            ls.append(dct)
            # print(ls)

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
        print(rec)
        for cam in cams:
            # i = str(self.cam_id[cam]+1)
            # imgname = os.path.join(self.folder, i+'/CAM'+i+'_000040.jpeg')
            imgname = rec['data'][cam]
<<<<<<< HEAD
            print(imgname)
            img = Image.open(imgname)
=======
            img = Image.open(imgname)#.convert('RGB')
>>>>>>> 51cf2c0a1849538467853557f1eb58641ee15710
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            #sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            #intrin = torch.Tensor(sens['camera_intrinsic'])
            #rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            #tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
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
            intrins.append(self.intrins[self.cam_index[cam]])
            rots.append(self.rots[self.cam_index[cam]])
            trans.append(self.trans[self.cam_index[cam]])
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        #print("Starting!!!")
        #print(intrins)
        #print(rots)
        #print(trans)
        #print(post_rots)
        #print(post_trans)

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
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class SimSegmentationData(SimData): #torch.utils.data.Dataset
    def __init__(self, *args, **kwargs):
        super(SimSegmentationData, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        rec = self.ixes[index]
        cams = self.choose_cams()
<<<<<<< HEAD
=======
        
>>>>>>> 51cf2c0a1849538467853557f1eb58641ee15710
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


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name, map_folder=''):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot, #os.path.join(dataroot, version),
                    verbose=False)
    nusc_maps = None
    if map_folder:
        nusc_maps = get_nusc_maps(map_folder)
    
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
        'multisegmentationdata': MultiSegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, nusc_maps=nusc_maps)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, nusc_maps=nusc_maps)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader

'''

if __name__=='__main__':
    dataroot='D:/dataset/nuscenes'
    version='mini'
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot, #os.path.join(dataroot, version),
                    verbose=False)
    map_folder='D:/dataset/nuscenes'

    H=900
    W=1600
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True

    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 6,
                }
    
    
    nusc_maps = get_nusc_maps(map_folder)

    gen = SimSegmentationData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf, nusc_maps=nusc_maps)


    imgs, rots, trans, intrins, post_rots, post_trans, binmap = gen.__getitem__(0)
    print(rots)
    print(trans)
    print(intrins)
    print(post_rots)
    print(post_trans)
'''