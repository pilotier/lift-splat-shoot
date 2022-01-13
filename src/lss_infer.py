import torch
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map, generate_video_from_imgs)
from .models import compile_model

# Add paths
dataroot = '/home/navya/data/LSS/LSSCAM'
map_folder = '/home/navya/data/nuscenes/trainval/v1.0-trainval'

modelf = '/home/navya/projects/lift-splat-shoot/model.pt'

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

cams = ['CAM_FRONT']

data_aug_conf = {
    'resize_lim': resize_lim,
    'final_dim': final_dim,
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': cams,
    'Ncams': 1,
}

# trainloader, valloader = compile_sim_data(version, dataroot, data_aug_conf=data_aug_conf,
#                                           grid_conf=grid_conf, bsz=bsz, nworkers=nworkers, length=max_frames)
# loader = trainloader if viz_train else valloader
nusc_maps = get_nusc_maps(map_folder)

device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

model = compile_model(grid_conf, data_aug_conf, outC=3)
print('loading', modelf)
model.load_state_dict(torch.load(modelf))
model.to(device)

dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
dx, bx = dx[:2].numpy(), bx[:2].numpy()



val = 0.01
fH, fW = final_dim
# fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
# gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
# gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

model.eval()
counter = 0



def infer_image(imgs):
    imgs
    rots = torch.tensor([[[ 0.8225,  0.0065,  0.5687],
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

    trans = torch.tensor([[ 1.5753,  0.5005,  1.5070],
                        [ 1.7220,  0.0048,  1.4949],
                        [ 1.5808, -0.4991,  1.5175],
                        [ 1.0485,  0.4831,  1.5621],
                        [ 0.0552,  0.0108,  1.5679],
                        [ 1.0595, -0.4672,  1.5505]])
        
    intrins = torch.tensor([[[1.2579e+03, 0.0000e+00, 8.2724e+02],
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
        
    post_rots = [[[1., 0., 0.],
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

    post_trans = [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]
    
    with torch.no_grad():
        out = model(imgs.to(device),
            rots.to(device),
            trans.to(device),
            intrins.to(device),
            post_rots.to(device),
            post_trans.to(device),
            )
        out = out.sigmoid().cpu()

    