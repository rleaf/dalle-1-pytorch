import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
# Prefer not using einsum for now
# from torch import einsum

class SimpleResBlock(nn.Module):
   def __init__(self, x):
      super().__init__()
      self.net = nn.Sequential(
         nn.Conv2d(x, x, 3, padding = 1),
         nn.ReLU(),
      )

   def forward(self, x):
      return self.net(x) + x

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
      return self.net(x) + x

class dVAE(nn.Module):
   def __init__(self, tokens, codebook_dim, hidden_dim, channels):
      super().__init__()

      # If stride = 2 for both enc/dec, log2(H or W) must be int. Asserted on line ~54 ballpark.
      self.encoder = nn.Sequential(
         # W' = (W - kernel + 2*padding) / stride + 1
         nn.Conv2d(channels, hidden_dim, 4, stride = 1, padding = 1),
         # SimpleResBlock(hidden_dim),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         # SimpleResBlock(hidden_dim),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         # SimpleResBlock(hidden_dim),
         nn.ReLU(),
         ResBlock(hidden_dim),
         ResBlock(hidden_dim),
         nn.Conv2d(hidden_dim, tokens, 1)
      )

      self.decoder = nn.Sequential(
         # W' = (W - 1)*stride - 2*padding + (kernel - 1) + 1
         nn.Conv2d(codebook_dim, hidden_dim, 1),
         ResBlock(hidden_dim),
         ResBlock(hidden_dim),
         nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         # SimpleResBlock(hidden_dim),
         nn.ReLU(),
         nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         # SimpleResBlock(hidden_dim),
         nn.ReLU(),
         nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         # SimpleResBlock(hidden_dim),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, channels, 1),
      )

      self.tokens = tokens
      # self.recon = F.mse_loss
      self.codebook = nn.Embedding(tokens, codebook_dim) # (tokens, dim)

   def hard_indices(self, images):
      # Circumvent gumbel sampling during training/testing 
      a = self(images, return_logits = True)  # (B, tokens, H', W')
      b = torch.argmax(a, dim = 1).flatten(1) # (B, H' * W')
      return b

   def codebook_decode(self, x):
      embeds = self.codebook(x)  # (B, H' * W', codebook_dim)
      b, n, d = embeds.shape
      hw = int(n ** (1/2))

      embeds = embeds.reshape(b, d, hw, hw)
      images = self.decoder(embeds)
      return images

   def forward(self, img, temp = 1.0, return_logits = False):  # (B, 3, H, W)
      """
      img  ->  enc(img)=logits  ->  gumbel(logits)=cont_one_hot  ->
      codebook(cont_one_hot)=tokens  -> 
      dec(tokens)=out  ->  ELBO(img,out)=loss
      """

      # assert log2(img.shape[0]).is_integer()
      logits = self.encoder(img) # (B, tokens, H', W')
      if return_logits:
         return logits

      # Disretizing logits 
      gumbel_logit = F.gumbel_softmax(logits.permute(0, 2, 3, 1), tau = temp) # (B, tokens, H', W') -> (B, H', W', tokens)
      # Training codebook
      tokens = torch.matmul(gumbel_logit, self.codebook.weight).permute(0, 3, 1, 2) # (B, H', W', dim) -> (B, dim, H', W')
      out = self.decoder(tokens) # (B, 3, H, W)

      recon_loss = F.mse_loss(img, out)

      logits = logits.permute(0, 2, 3, 1).flatten(start_dim = 1, end_dim = -2) # (B, H', W', tokens) -> (B, H' * W', tokens)
      logits = F.log_softmax(logits, dim = -1)
      log_uniform = torch.log(torch.tensor([1. / self.tokens], device = logits.device))
      kl_div = F.kl_div(log_uniform, logits, None, None, 'batchmean', log_target = True) # DKL(q_\phi || p_\psi)

      loss = recon_loss + kl_div

      return loss, out


# torch.manual_seed(0)
# model = dVAE(512, 128, 24, 1)
# input = torch.rand((20, 1, 28, 28))
# loss, out = model(input, temp = 1.)
# print(loss, out.shape, 'toad')
# # j = model.hard_indices(input)
# # y = model.codebook_decode(j)
# # print(y.shape, 'toad')
