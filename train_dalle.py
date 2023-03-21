import code
import os
import argparse
import math

# wandb
import wandb

# torch
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.datasets as dset
import torchvision.transforms as T

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# files
from models.dvae import dVAE

def main(config):
   pass

if __name__ == '__main__':
   args = argparse.ArgumentParser()

   args.add_argument('--epochs', type=int, default = 10)

   config = args.parse_args()

   main(config)