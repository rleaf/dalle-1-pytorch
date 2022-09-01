import torch
import torch.nn as nn
import torch.nn.functional as F
# Prefer not using einsum for now
# from torch import einsum

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
      # Circumvent gumbel sampling during testing & eventual joint training ? 
      # a = self(images, return_logits = True)
      # b = torch.argmax(a, dim = 1)
      # return b
      # If I do this, do I need a decode() method to handle b and feed it through: codebook(b) = c --> self.decoder(c) = d? 
      pass


   def forward(self, img, return_logits = False):  # (B, 3, H, W)
      """
      img -> enc(img) = logits -> gumbel(logits) = cont_one_hot
      codebook(cont_one_hot) = tokens -> 
      dec(tokens) -> out -> ELBO(img, out) = loss
      """
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

      recon_loss = F.mse_loss(img, out)
      latent = latent.permute(0, 3, 2, 1)
      logits = F.log_softmax(latent, dim = -1)
      log_uniform = torch.log(torch.tensor([1. / self.tokens], device = logits.device))
      kl_div = F.kl_div(log_uniform, logits, None, None, 'batchmean', log_target = True)

      loss = recon_loss + kl_div

      return loss, out


torch.manual_seed(0)
model = dVAE(200, 512)
input = torch.rand((20, 3, 256, 256))
loss = model(input)
print(loss.shape, 'toad')
