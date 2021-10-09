FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt update -y
RUN apt install gcc -y
RUN pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
RUN apt install ffmpeg libsm6 libxext6  -y

ADD . /workspace 