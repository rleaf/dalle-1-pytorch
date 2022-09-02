import os
import argparse

# torch
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
   opt = Adam(dvae.parameters(), lr = learning_rate)

   if torch.cuda.is_available():
      dvae.cuda()

   mnist_train = dset.MNIST('./MNIST_data' , train=True, download=True,
                            transform=T.ToTensor())
   loader_train = DataLoader(mnist_train, batch_size=batch_size,
                             shuffle=True, drop_last=True, num_workers=2)

   for epoch in range(epochs):
      dvae.train()

      for i, (data, labels) in enumerate(loader_train):
         # data torch.size([batch_size, 1, 28, 28])
         print('Index: {}'.format(i))
         data = data.to(device=next(dvae.parameters()).device)
         loss, out = dvae(data)
         opt.zero_grad()
         loss.backward()
         opt.step()

      print('Epoch: {} \tLoss: {:.6f}'.format(
         epoch, loss.data))

      if i & 100 == 0:
         j = 4

      # Reconstruction via argmax > gumbel softmax
      #    with torch.no_grad():
            # logits = dvae.hard_indices(data[:j])
            # hard_out = dvae.codebook_decode(logits) # shape (j, C, H, W) probably

         plt.figure(figsize=(10, 1))
         gspec = gridspec.GridSpec(1, 10)
         gspec.update(wspace=0.05, hspace=0.05)
         for i, sample in enumerate(out[:j]):
            ax = plt.subplot(gspec[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
            plt.savefig(os.path.join(config.output, 'vae_generation.jpg'))

         break




   torch.save(dvae, os.path.join('./', 'dvae_weights.pt'))



if __name__ == '__main__':
   args = argparse.ArgumentParser()
   
   args.add_argument('--epochs', type = int, default = 10)
   args.add_argument('--batch_size', type = int, default = 128)
   # args.add_argument('--image_size', type = int, default = 28) # MNIST
   # args.add_argument('--image_path', type = str, default = './')
   args.add_argument('--num_tokens', type = int, default = 128)
   args.add_argument('--codebook_dim', type = int, default = 128)
   args.add_argument('--hidden_dim', type = int, default = 24)
   args.add_argument('--learning_rate', type = float, default = 1e-3)
   args.add_argument('--channels', type = int, default = 1) # MNIST is b/w

   config = args.parse_args()

   main(config)