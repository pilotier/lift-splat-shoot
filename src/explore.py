"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

from torchvision.utils import save_image

import os

from .data import compile_sim_data
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map, generate_video_from_imgs)
from .models import compile_model

# 0 = img_vehicle, 1 = img_road_segment, 2 = img_lane_divider
def sim_model_preds(version,
                    modelf,
                    dataroot='/home/navya/data/sim/LSS/LSSCAM',
                    map_folder='/home/navya/data/nuscenes/trainval/',
                    gpuid=0,
                    viz_train=False,
                    video_output=True,
                    max_frames=-1,
                    channel=0,

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    bsz=1,
                    nworkers=0, #10
                    ):
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
    trainloader, valloader = compile_sim_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers, length=max_frames)
    loader = trainloader if viz_train else valloader
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
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    try:
        os.mkdir('./output')
    except OSError as error:
        pass

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binmaps) in enumerate(loader):
            out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            out = out.sigmoid().cpu()
            print(imgs.shape[0])
            for si in range(imgs.shape[0]):
                # print(si)
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    # print(imgi,img)
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                ax = plt.subplot(gs[0, 0:2])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                # Makes the background black
                # ax.add_patch(mpatches.Rectangle((0, 0), out.shape[3], out.shape[3], facecolor='k'))

                plt.setp(ax.spines.values(), color='b', linewidth=0)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Map Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 1.00, 0, 1.0), label='Groundtruth Map')
                ], loc=(0.01, 0.86), labelcolor='k')
                
                # TO ADD TO THE LEFT SIDE ALSO
                plt.imshow(binmaps[si,channel], vmin=0, vmax=1, cmap='Blues')
                # plt.imshow(out[si,channel], vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)
                # rec = loader.dataset.ixes[counter]
                # plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                ax = plt.subplot(gs[0, 2])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=0)
                plt.imshow(out[si,channel], vmin=0, vmax=1, cmap='Blues')
                ##plt.imshow(out[si,2], vmin=0, vmax=1, cmap='Greens', alpha=0.4)
                # plt.imshow(out[si,0], vmin=0, vmax=1, cmap='Reds')
                # plt.imshow(out[si,1], vmin=0, vmax=1, cmap='Blues')
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig("output/"+imname)
                counter += 1

            if (counter > max_frames and max_frames > 0):
                print("BROKEN")
                break
    
    # if video_output:
    #     generate_video_from_imgs('output', '.jpg')