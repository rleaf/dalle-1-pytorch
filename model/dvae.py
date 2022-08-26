import torch
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

      self.codebook = nn.Embedding(tokens, dim) # (tokens, dim)

   def forward(self, x):  # (B, 3, H, W)
      logits = self.encoder(x) # (B, tokens, H', W')
      gumbel = F.gumbel_softmax(logits, tau=1.).permute(0, 2, 3, 1) # (B, tokens, H', W') -> (B, H', W', tokens)
      print('shapes', self.codebook, gumbel.shape)
      # sampled = self.codebook(gumbel.to(dtype=torch.long))
      # I want to train a model to use a lookup table, not use a lookup table myself
      sampled = torch.matmul(gumbel, self.codebook.weight).permute(0, 3, 1, 2) # (B, H', W', dim) -> (B, dim, H', W')
      out = self.decoder(sampled)

      return out