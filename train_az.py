import sys
import src

def az_multi_train():
    data_folder = sys.argv[1]
    maps_folder = sys.argv[2]
    mounted_output_path = sys.argv[3]
    src.train.multi_train('mini', dataroot=data_folder, map_folder=maps_folder, gpuid=-1, output_folder=mounted_output_path)
    print("success")

az_multi_train()