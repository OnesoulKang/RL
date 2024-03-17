import torch
import torch.multiprocessing as mp

import pybullet as pb

def mp_print(rank):
    print("rank :", rank)
    c = pb.connect(pb.GUI)

if __name__ == '__main__':
    num_process = 4
    processes = []

    for rank in range(num_process):
        p = mp.Process(target=mp_print, args=(rank, ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()