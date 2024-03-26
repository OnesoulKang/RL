import os
import torch
import argparse
import random
import numpy as np

def load(save_path, net):
    files = os.listdir(save_path)
    files = [file for file in files if '.pt' in file]
    files_dict = {}
    for file in files:
        files_dict[file] = int(file.split('.')[0])
    files_dict = sorted(files_dict, key=lambda x: files_dict[x])
    
    load_file = save_path+"/"+files_dict[-1]
    print("Load file:", load_file)
    load_file = torch.load(load_file)

    net.load_state_dict(load_file['model_state_dict'])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Walker2d-v4')
    parser.add_argument('--play', default=False)
    parser.add_argument('--imitation', default=False)
    parser.add_argument('--seed', default=1)
    parser.add_argument('--imitation_coef', default=0.1)

    return parser.parse_args()


def set_seed(seed):
    if seed==-1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)