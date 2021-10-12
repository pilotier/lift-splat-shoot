import sys, os
import subprocess
import src

def az_multi_train():
    # unzip
    bashCommand = "mkdir TEST && cd ./mini && mkdir HELLO"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)

    data_folder = 'mini'
    maps_folder = 'maps'
    mounted_output_path = sys.argv[3]
    print(data_folder)
    print(os.listdir('./'))
    print(os.listdir('./'+data_folder))
    #src.train.multi_train('mini', dataroot=data_folder, map_folder=maps_folder, gpuid=-1, output_folder=mounted_output_path)
    print("success")
    print(os.listdir('./'))

az_multi_train()