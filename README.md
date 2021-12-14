#  File "/home/navya/anaconda3/lib/python3.9/site-packages/efficientnet_pytorch/utils.py", line 271, in forward
#    x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#RuntimeError: Expected 4-dimensional input for 4-dimensional weight [32, 3, 3, 3], but got 5-dimensional input of size [4, 1, 3, 130, 354] instead
# Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D

PyTorch code for Lift-Splat-Shoot (ECCV 2020).

**Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D**  
Jonah Philion, [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)\
ECCV, 2020 (Poster)\
**[[Paper](https://arxiv.org/abs/2008.05711)] [[Project Page](https://nv-tlabs.github.io/lift-splat-shoot/)] [[10-min video](https://youtu.be/oL5ISk6BnDE)] [[1-min video](https://youtu.be/ypQQUG4nFJY)]**

**Abstract:**
The goal of perception for autonomous vehicles is to extract semantic representations from multiple sensors and fuse these representations into a single "bird's-eye-view" coordinate frame for consumption by motion planning. We propose a new end-to-end architecture that directly extracts a bird's-eye-view representation of a scene given image data from an arbitrary number of cameras. The core idea behind our approach is to "lift" each image individually into a frustum of features for each camera, then "splat" all frustums into a rasterized bird's-eye-view grid. By training on the entire camera rig, we provide evidence that our model is able to learn not only how to represent images but how to fuse predictions from all cameras into a single cohesive representation of the scene while being robust to calibration error. On standard bird's-eye-view tasks such as object segmentation and map segmentation, our model outperforms all baselines and prior work. In pursuit of the goal of learning dense representations for motion planning, we show that the representations inferred by our model enable interpretable end-to-end motion planning by "shooting" template trajectories into a bird's-eye-view cost map output by our network. We benchmark our approach against models that use oracle depth from lidar. Project page: [https://nv-tlabs.github.io/lift-splat-shoot/](https://nv-tlabs.github.io/lift-splat-shoot/).

**Questions/Requests:** Please file an [issue](https://github.com/nv-tlabs/lift-splat-shoot/issues) if you have any questions or requests about the code or the [paper](https://arxiv.org/abs/2008.05711). If you prefer your question to be private, you can alternatively email me at jphilion@nvidia.com.

### Citation
If you found this codebase useful in your research, please consider citing
```
@inproceedings{philion2020lift,
    title={Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D},
    author={Jonah Philion and Sanja Fidler},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2020},
}
```

### Preparation
Download nuscenes data from [https://www.nuscenes.org/](https://www.nuscenes.org/). Install dependencies.

```
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

#### Simple Install with Anaconda
Simply run the following command to create the conda virtual environment (tested on Windows 10).
```
conda env create -f conda_env.yaml
```
The current LST version of pyTorch is known to work, which can be installed:
```
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

#### Simple Install with Docker
Simply run the following command to download the docker image, after installing docker on your system.
```
docker pull pilotier/lift-splat-shoot:latest
```
Then, to run the docker image, enter:
```
docker run -it --name NAME -v PATH/TO/NUSCENE_ROOT:/dataset:ro pilotier/lift-splat-shoot:latest
```
where NAME is the name of your docker image, and PATH/TO/NUSCENE_ROOT is the directory that contains the dataset (in this case the nuscenes folder, which contains 'maps' and 'mini').

#### Correcting nuScenes Data Path
The DATAROOT folder is simply the nuscenes folder which contains different version (in our case "mini") folders.

To get the map visualization working, the Map Extension pack (currently v1.3) needs to be downloaded. The maps folder should be placed in the same DATAROOT folder for nuscenes (besides "mini").

The overall structure of the directory will look as follows:

-> nuscenes (_or whatever name you call it_)\
&nbsp;&nbsp;&nbsp;&nbsp;|__ mini\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ samples\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ sweeps\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ maps (_has 4 images_)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ v1.0-mini\
&nbsp;&nbsp;&nbsp;&nbsp;|__ trainval\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ ...\
&nbsp;&nbsp;&nbsp;&nbsp;|__ test\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ ...\
&nbsp;&nbsp;&nbsp;&nbsp;|__ maps (_from map extentions v1.3_)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ basemap\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ expansion\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ prediction\
&nbsp;


Finally, the directory structure passed to the program will simply be _/path/to/nuscenes_, for both DATAROOT and MAP_FOLDER.


### Pre-trained Model
Download a pre-trained BEV vehicle segmentation model from here: [https://drive.google.com/file/d/18fy-6beTFTZx5SrYLs9Xk7cY-fGSm7kw/view?usp=sharing](https://drive.google.com/file/d/18fy-6beTFTZx5SrYLs9Xk7cY-fGSm7kw/view?usp=sharing) 

| Vehicle IOU (reported in paper)        | Vehicle IOU (this repository)         |
|:-------------:|:-------------:| 
| 32.07      | 33.03 |

### Evaluate a model
Evaluate the IOU of a model on the nuScenes validation set. To evaluate on the "mini" split, pass `mini`. To evaluate on the "trainval" split, pass `trainval`.

```
python main.py eval_model_iou mini/trainval --modelf=MODEL_LOCATION --dataroot=NUSCENES_ROOT
```

### Visualize Predictions
Visualize the BEV segmentation output by a model:

```
python main.py viz_model_preds mini/trainval --modelf=MODEL_LOCATION --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
```
<img src="./imgs/eval.gif">

### Visualize Input/Output Data (optional)
Run a visual check to make sure extrinsics/intrinsics are being parsed correctly. Left: input images with LiDAR scans projected using the extrinsics and intrinsics. Middle: the LiDAR scan that is projected. Right: X-Y projection of the point cloud generated by the lift-splat model. Pass `--viz_train=True` to view data augmentation.

```
python main.py lidar_check mini/trainval --dataroot=NUSCENES_ROOT --viz_train=False
```
<img src="./imgs/check.gif">

### Train a model (optional)
Train a model. Monitor with tensorboard.

```
python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
tensorboard --logdir=./runs --bind_all
```

### Acknowledgements
Thank you to Sanja Fidler, as well as David Acuna, Daiqing Li, Amlan Kar, Jun Gao, Kevin, Xie, Karan Sapra, the NVIDIA AV Team, and NVIDIA Research for their help in making this research possible.
