import os
import argparse

# torch

import torch
from torch.optim import Adam
# from torch.utils.data import DataLoader
# import torchvision.datasets as dset
# import torchvision.transforms as T

# files

from dvae import dVAE

def main(config):
   pass


if __name__ == '__main__':
   args = argparse.ArgumentParser()
   
   args.add_argument('--hparam', type = int, default = 0)

   config = args.parse_args()

   main(config)