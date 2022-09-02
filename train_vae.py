import os
import argparse

# torch
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T

# files
from model.dvae import dVAE


def train_vae(epoch, model, train_loader, lr):
   model.train()
   train_loss = 0
   # num_classes = 10
   optimizer = Adam(model.parameters(), lr = lr)

   for batch_idx, (data, labels) in enumerate(train_loader):
      data = data.to(device=next(model.parameters()).device)
      loss, _ = model(data)
      optimizer.zero_grad()
      loss.backward()
      train_loss += loss.data
      optimizer.step()

   print('Train Epoch: {} \tLoss: {:.6f}'.format(
       epoch, loss.data))

def main(config):
   epochs = config.epochs
   batch_size = config.batch_size
   # image_size = config.image_size
   # image_path = config.image_path

   num_tokens = config.num_tokens
   codebook_dim = config.codebook_dim
   hidden_dim = config.hidden_dim
   learning_rate = config.learning_rate
   channels = config.channels

   dvae = dVAE(num_tokens, codebook_dim, hidden_dim, channels)
   # opt = Adam(dvae.parameters(), lr = learning_rate)

   if torch.cuda.is_available():
      dvae.cuda()

   mnist_train = dset.MNIST('./MNIST_data' , train=True, download=True,
                            transform=T.ToTensor())
   loader_train = DataLoader(mnist_train, batch_size=batch_size,
                             shuffle=True, drop_last=True, num_workers=2)

   for epoch in range(0, epochs):
      train_vae(epoch, dvae, loader_train, learning_rate)

   torch.save(dvae, os.path.join('./', 'dvae_weights.pt'))



if __name__ == '__main__':
   args = argparse.ArgumentParser()
   
   args.add_argument('--epochs', type = int, default = 10)
   args.add_argument('--batch_size', type = int, default = 128)
   # args.add_argument('--image_size', type = int, default = 28) # MNIST
   # args.add_argument('--image_path', type = str, default = './')
   args.add_argument('--num_tokens', type = int, default = 512)
   args.add_argument('--codebook_dim', type = int, default = 512)
   args.add_argument('--hidden_dim', type = int, default = 64)
   args.add_argument('--learning_rate', type = float, default = 1e-3)
   args.add_argument('--channels', type = int, default = 1) # MNIST is b/w

   config = args.parse_args()

   main(config)
