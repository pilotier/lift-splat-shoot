import torch
from torchvision import transforms
import numpy as np
import os
import io
import glob
from matplotlib import pyplot as plt
import time
from PIL import Image

from tools import normalize_img, gen_dx_bx, img_transform
from models import compile_model

# Add model weight path
modelf = '/Users/navyarao/Desktop/projects/lift-splat-shoot/weights/model300.pt'
input_path = '/Users/navyarao/Desktop/projects/sim/LSSCAM/1/*.jpeg'
output_path = '/Users/navyarao/Desktop/projects/lift-splat-shoot/output/'

# Initialization
gpuid=0
viz_train=False
video_output=True
max_frames=-1
channel=1
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

bsz=1
nworkers=0
    
grid_conf = {
    'xbound': xbound,
    'ybound': ybound,
    'zbound': zbound,
    'dbound': dbound,
}

data_aug_conf = {
    'resize_lim': resize_lim,
    'final_dim': final_dim,
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
}

def sample_augmentation():
    H, W = data_aug_conf['H'], data_aug_conf['W']
    fH, fW = data_aug_conf['final_dim']
    resize = max(fH/H, fW/W)
    resize_dims = (int(W*resize), int(H*resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim']))*newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    return resize, resize_dims, crop, flip, rotate

device = torch.device('cpu') #if gpuid < 0 else torch.device(f'cuda:{gpuid}')

model = compile_model(grid_conf, data_aug_conf, outC=3)
print('loading', modelf)
model.load_state_dict(torch.load(modelf, map_location=device))
model.to(device)

#Ego pose
dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
dx, bx = dx[:2].numpy(), bx[:2].numpy()


# Should ideally be getting all this values for sim camera

rots = []
rot = torch.tensor([[ 0.0103,  0.0084,  0.9999],
                    [-0.9999,  0.0123,  0.0102],
                    [-0.0122, -0.9999,  0.0086]])
rots.append(rot) 
rots = torch.stack(rots)

tran1 = []
trans = []
tran = torch.tensor([ 1.7220,  0.0048,  1.4949])
tran1.append(tran)
tran1 = torch.stack(tran1)
trans.append(tran1)
trans = torch.stack(trans)

intrins = []
intrin = torch.tensor([[1.2528e+03, 0.0000e+00, 8.2659e+02],
                    [0.0000e+00, 1.2528e+03, 4.6998e+02],
                    [0.0000e+00, 0.0000e+00, 1.0000e+00]])
intrins.append(intrin)
intrins = torch.stack(intrins)

# Image load and operations from here
def process_image_data(input_image):
    imgs = []
    img1 = []
    post_rots = []
    post_trans = []
    post_trans1 = []
    post_rot = torch.eye(2)
    post_tran = torch.zeros(2)

    img = Image.open(input_image)
    # img = np.array(img)
    # img = img[:,:,:3]
    # img = Image.fromarray(img)
    resize, resize_dims, crop, flip, rotate = sample_augmentation()
    img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                resize=resize,
                                                resize_dims=resize_dims,
                                                crop=crop,
                                                flip=flip,
                                            rotate=rotate,
                                                )

    img1.append(normalize_img(img))
    img1 = torch.stack(img1)
    imgs.append(img1)
    imgs = torch.stack(imgs)

    post_rot = torch.eye(3)
    post_rot[:2, :2] = post_rot2
    post_rots.append(post_rot)
    post_rots = torch.stack(post_rots)

    post_tran = torch.zeros(3)
    post_tran[:2] = post_tran2
    post_trans1.append(post_tran)
    post_trans1 = torch.stack(post_trans1)
    post_trans.append(post_trans1)
    post_trans = torch.stack(post_trans)

    model.eval()
    with torch.no_grad():
        out = model(imgs.to(device),
            rots.to(device),
            trans.to(device),
            intrins.to(device),
            post_rots.to(device),
            post_trans.to(device),
            )
        out = out.sigmoid().cpu()

    

    img_final = torch.cat((out[0,0],out[0,1],out[0,2]),1)
    # print(img_final.shape)
    
    pil_image = transforms.ToPILImage()(img_final).convert("RGB")
    
    return pil_image
    

imgs = glob.glob(input_path)
for img in imgs:
    print(img)
    name = output_path + img.split('/')[-1].split('.')[0] + '.png'
    bev_image = process_image_data(img)
    bev_image.save(name)




# print(out[0,0])
# print(out[0,1])
# print(out[0,2])
# out[0,0] *= (255.0/out[0,0].max())
# out[0,1] *= (255.0/out[0,1].max())
# out[0,2] *= (255.0/out[0,2].max())
# print("After")
# print(out[0,0])
# print(out[0,1])
# print(out[0,2])
# name = output_path + input_image.split('/')[-1].split('.')[0] + '.png'
# plt.subplot(1,3,1)
# plt.imshow(out[0,0])
# plt.subplot(1,3,2)
# plt.imshow(out[0,1])
# plt.subplot(1,3,3)
# plt.imshow(out[0,2])
# plt.savefig(name)
# print(out[0,0].shape)