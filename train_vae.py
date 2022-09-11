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
import torchvision.datasets as dset
import torchvision.transforms as T

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# files
from model.dvae import dVAE


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
   # output = config.output

   dvae = dVAE(num_tokens, codebook_dim, hidden_dim, channels)
   opt = Adam(dvae.parameters(), lr = learning_rate)

   dvae_config = {
      'hparams': 'copying from Hannu',
      'epochs': epochs,
      'learning_rate': learning_rate,
      'batch_size': batch_size,
      'num_tokens': num_tokens,
      'codebook_dim': codebook_dim,
      'hidden_dim': hidden_dim,
   }

   wandb.init(
      project = 'Small DALL-E',
      config = dvae_config)

   if torch.cuda.is_available():
      dvae.cuda()

   # fshn_mnist_train = dset.FashionMNIST('./fashion_MNIST', train=True, download=True, transform=T.ToTensor())
   mnist_train = dset.MNIST('./MNIST_data' , train=True, download=True, transform=T.ToTensor())
   loader_train = DataLoader(mnist_train, batch_size=batch_size,
                             shuffle=True, drop_last=True, num_workers=0)

   temp = 1.0
   step = 0
   plt.figure(figsize=(10, 1))

   with open('{}/config.txt'.format('./dvae_generation'), 'w') as f:
      for i in vars(config):
         f.write('{} = {} \n'.format(i, getattr(config, i)))

   for epoch in range(epochs):
      dvae.train()
      j = 10

      for i, (data, labels) in enumerate(loader_train):
         data = data.to(device=next(dvae.parameters()).device)
         loss, out = dvae(data, temp)
         # print('Index: {}'.format(i))
         opt.zero_grad()
         loss.backward()
         opt.step()

         logs = {}

         if i % 100 == 0:
            print('iter {}/{}'.format(i, len(loader_train)))
            with torch.no_grad():
               codes = dvae.hard_indices(data[:j])
               hard_recons = dvae.codebook_decode(codes)

            hard_recons, codes = map(lambda x: x.detach().cpu(), (hard_recons, codes))
            # out_np = out.cpu().detach()
            # data2 = data.cpu().detach()

            logs = {
               **logs, 
               'sample images': wandb.Image(data[:j], caption = 'original image'),
               'reconstructed images': wandb.Image(hard_recons, caption = 'reconstructed image'),
               'codebook_indices': wandb.Histogram(codes),
            }

            gspec = gridspec.GridSpec(1, 10)
            gspec.update(wspace=0.05, hspace=0.05)
            for k, sample in enumerate(hard_recons[:j]):
               ax = plt.subplot(gspec[k])
               plt.axis('off')
               ax.set_xticklabels([])
               ax.set_yticklabels([])
               ax.set_aspect('equal')
               plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
            plt.savefig(os.path.join('./dvae_generation', 'dvae_generationE{}.jpg'.format(epoch)))

         temp = max(temp * math.exp(-1e-6 * step), 0.5)

         step += 1

         logs = {
            **logs,
            'iteration': i,
            'loss': loss,
            'temp': temp
         }

         wandb.log(logs)
         
      print('Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data))

   torch.save(dvae, os.path.join('./', 'dvae_weights.pt'))

if __name__ == '__main__':
   args = argparse.ArgumentParser()
   
   args.add_argument('--epochs', type = int, default = 10)
   args.add_argument('--batch_size', type = int, default = 128)
   # args.add_argument('--image_size', type = int, default = 28) # MNIST
   # args.add_argument('--image_path', type = str, default = './')
   args.add_argument('--num_tokens', type = int, default = 256)
   args.add_argument('--codebook_dim', type = int, default = 256)
   args.add_argument('--hidden_dim', type = int, default = 128)
   args.add_argument('--learning_rate', type = float, default = 1e-3)
   args.add_argument('--channels', type = int, default = 1) # MNIST is b/w
   # args.add_argument('--output', type = str, default = './dvae_generation')

   config = args.parse_args()

   main(config)
