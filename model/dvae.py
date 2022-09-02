import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
# Prefer not using einsum for now
# from torch import einsum

class dVAE(nn.Module):
   def __init__(self, tokens, codebook_dim, hidden_dim, channels):
      super().__init__()

      # If stride = 2 for both enc/dec, log2(H or W) must be int. Asserted on line ~54 ballpark.
      self.encoder = nn.Sequential(
         # W' = (W - kernel + 2*padding) / stride + 1
         nn.Conv2d(channels, hidden_dim, 4, stride = 1, padding = 1),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, tokens, 1)
      )

      self.decoder = nn.Sequential(
         # W' = (W - 1)*stride - 2*padding + (kernel - 1) + 1
         nn.ConvTranspose2d(codebook_dim, hidden_dim, 4, stride = 1, padding = 1),
         nn.ReLU(),
         nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         nn.ReLU(),
         nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride = 1, padding = 1),
         nn.ReLU(),
         nn.Conv2d(hidden_dim, channels, 1),
      )

      self.tokens = tokens
      # self.recon = F.mse_loss
      self.codebook = nn.Embedding(tokens, codebook_dim) # (tokens, dim)

   def hard_indices(self, images):
      # Circumvent gumbel sampling during training/testing 
      a = self(images, return_logits=True)  # (B, tokens, H', W')
      b = torch.argmax(a, dim = 1).flatten(1) # (B, tokens * H' * W')
      return b
      # If I do this, do I need a decode() method to handle b and feed it through: codebook(b) = c --> self.decoder(c) = d? 

   def codebook_decode(self, x):
      pass


   def forward(self, img, return_logits = False):  # (B, 3, H, W)
      """
      img -> enc(img) = logits -> gumbel(logits) = cont_one_hot
      codebook(cont_one_hot) = tokens -> 
      dec(tokens) -> out -> ELBO(img, out) = loss
      """
      # assert log2(img.shape[0]).is_integer()

      logits = self.encoder(img) # (B, tokens, H', W')

      if return_logits:
         return logits

      cont_one_hot = F.gumbel_softmax(logits, dim = 1, tau = 1.).permute(0, 2, 3, 1) # (B, tokens, H', W') -> (B, H', W', tokens)
      # cont_one_hot = cont_one_hot.flatten(1) # (B, H' * W' * tokens)
      # tokens = self.codebook(cont_one_hot.long()) 
      # n = torch.tensor(tokens.shape[-1])
      # h = w = int(torch.sqrt(n))
      # tokens = tokens.reshape()

      tokens = torch.matmul(cont_one_hot, self.codebook.weight).permute(0, 3, 1, 2) # (B, H', W', dim) -> (B, dim, H', W')
      # test2 = einsum('b h w n, n d -> b d h w', cont_one_hot, self.codebook.weight)
      out = self.decoder(tokens) # (B, 3, H, W)

      # print(img.shape, logits.shape, out.shape, 'toads2')
      recon_loss = F.mse_loss(img, out)
      logits = logits.permute(0, 2, 3, 1) # (B, H', W', tokens)
      logits = F.log_softmax(logits, dim = -1)
      log_uniform = torch.log(torch.tensor((1. / self.tokens), device = logits.device))
      kl_div = F.kl_div(log_uniform, logits, None, None, 'batchmean', log_target = True) # DKL(q_\phi || p_\psi)

      loss = recon_loss + kl_div

      return loss, out


# torch.manual_seed(0)
# model = dVAE(200, 512, 64, 1)
# input = torch.rand((20, 1, 28, 28))
# loss, out = model(input)
# print(loss, out.shape, 'toad')
