import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

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
         nn.ConvTranspose2d(dim, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
         nn.ReLU(),
         nn.Conv2d(64, 3, 1)
      )

      self.tokens = tokens
      # self.recon = F.mse_loss
      self.codebook = nn.Embedding(tokens, dim) # (tokens, dim)

   def codebook_indices(self, x):
      # Circumvent gumbel sampling during joint training? 
      # a = self(images, return_latent = True)
      # b = torch.argmax(a, dim = ...)
      # return b
      # If I do this, do I need a decode() method to handle b and feed it through: codebook(b) = c --> self.decoder(c) = d? 
      pass


   def forward(self, img, return_latent = True):  # (B, 3, H, W)
      """
      img -> enc(img) = image_tokens ->
      codebook(image_tokens) = logits ->
      dec(logits) -> out -> ELBO(img, out) = loss
      """
      latent = self.encoder(img) # (B, tokens, H', W')

      if return_latent:
         return latent

      cont_one_hot = F.gumbel_softmax(latent, dim = 1, tau = 1.).permute(0, 2, 3, 1) # (B, tokens, H', W') -> (B, H', W', tokens)
      tokens = torch.matmul(cont_one_hot, self.codebook.weight).permute(0, 3, 1, 2) # (B, H', W', dim) -> (B, dim, H', W')
      test2 = einsum('b h w n, n d -> b d h w', cont_one_hot, self.codebook.weight)
      print(test2.shape, tokens.shape, 'here it is')
      out = self.decoder(tokens) # (B, 3, H, W)


      # # recon_loss = self.recon(img, out)
      recon_loss = F.mse_loss(img, out)
      latent = latent.permute(0, 3, 2, 1)
      logits = F.log_softmax(latent, dim = -1)
      log_uniform = torch.log(torch.tensor([1. / self.tokens], device = logits.device))
      kl_div = F.kl_div(log_uniform, logits, None, None, 'batchmean', log_target = True)

      loss = recon_loss + kl_div

      return loss


torch.manual_seed(0)
model = dVAE(200, 512)
input = torch.rand((20, 3, 256, 256))
loss = model(input)
print(loss)
