import torch.nn as nn
import torch.nn.functional as F

class dVAE(nn.Module):
   def __init__(self, tokens, dim):
      super().__init__()

      self.encoder = nn.Sequential(
         nn.Conv2d(3, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.Conv2d(64, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.Conv2d(64, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.Conv2d(64, tokens, 1)
      )

      self.decoder = nn.Sequential(
         # nn.Conv2d(dim, 64, 4, stride = 2, padding = 1)
         nn.ConvTranspose2d(dim, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.Conv2d(64, 3, 1)
      )

      self.codebook = nn.Embedding(tokens, dim)

   # def forward(self, x):
   #    logits = self.encoder(x)
   #    gumbel = F.gumbel_softmax(logits, tau = 1.)




      