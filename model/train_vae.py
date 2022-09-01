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
   epochs = config.epochs
   batch_size = config.batch_size
   image_size = config.image_size
   image_path = config.image_path

   tokens = config.num_tokens
   codebook_dim = config.codebook_dim
   hidden_dim = config.hidden_dim


if __name__ == '__main__':
   args = argparse.ArgumentParser()
   
   args.add_argument('--epochs', type = int, default = 10)

   config = args.parse_args()

   main(config)