import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, get_nusc_maps, get_local_map

class Sim:
	def __init__(self, dataroot: str):
		self.dataroot = dataroot

class SimData(torch.utils.data.Dataset):
	def __init__(self, sim, is_train, data_aug_conf, grid_conf):
		self.sim = sim
		self.is_train = is_train
		self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()


    def prepro(self):
    	'''
    	Should return a list of samples containing key-value pairs of data
		Ex: [{'timestamp': , 'data':{'CAM_FRONT':,'CAM_FRONT_LEFT':, ....}},{},...]
    	'''

    	return

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
        	'''
			Change images path, sensor data

        	'''
            # samp = self.nusc.get('sample_data', rec['data'][cam])
            # imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            # intrin = torch.Tensor(sens['camera_intrinsic'])
            # rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            # tran = torch.Tensor(sens['translation'])

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
    

    def get_binmap(self, rec):
    	'''
    	Load binmaps directly from sim data
    	'''
    	return


    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']

        return cams

    def __str__(self):
        return f"""Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class MultiSegmentationSimData(SimData):
	'''
	Datasets for the loader
	'''
	def __init__(self, *args, **kwargs):
		super(MultiSegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        #binimg = self.get_binimg(rec)
        binmap = self.get_binmap(rec)

        # print("rec: ", rec)
        # print("imgs",imgs,"rots", rots,"trans", trans,"intrins", intrins, "post_rots", post_rots,"post_trans", post_trans)
        # print("binmaps",binmap)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binmap

def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_sim_data(dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers):
    '''
    Create dataset for sim data.
    Use torch.utils.data.Dataloader to load it up from datasets

    '''
    sim = Sim(dataroot)
    parser = {
        'simdata' : MultiSegmentationSimData
    }[simdata]

    simdata = parser(sim, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    simloader = torch.utils.data.DataLoader(simdata, batch_size=bsz, shuffle=False, num_workers=nworkers)

    return simloader
