import torch
import torch.nn as nn

from transformer import Transformer
from dvae import dVAE

class DALLE(nn.Module):
   """
   INP  ->  enc(INP)=a  ->  codebook(a)=b  ->
   preprocess(b)=c  ->  transformer(c)=d  ->
   (transformer(d)=d* x token_len)=d2  ->  dec(d2)=OUT

   preprocess() {
      - "BPE" input text sequence
      - return concat text seq w/ codebook tokens
   }

   In training the Transformer autoregressively generates the next token 
   to represent the masked section of an image with the given text seq & codebook tokens
   provided as inp.
      
   Stage 1:
      Train dVAE on images where kl divergence in variational lower bound
      measures from auxiliary $q_phi$ to uniform categorical $p_psi$ later learned
      by the transformer.

   Stage 2:
      Continue to optimize variational lower bound wrt $p_psi$.
      

   """
   def __init__(
      self,
      vae,
      dim,
      depth,
      head_dim,
      heads,
      ff_dim,
      dropout = 0.0,
      ff_dropout = 0.0,
   ):
      super().__init__()

      self.vae = vae

      self.transformer = Transformer(
         dim = dim,
         depth = depth,
         head_dim = head_dim,
         heads = heads,
         ff_dim = ff_dim,
         dropout = dropout,
         ff_dropout = ff_dropout
      )

   
   def forward(self, text, image):
      """
      *enc(inp)=a  ->  prep(a)=b  ->
      transformer(b)=c  ->  ce_loss(c, label)

      prep(): same as above
      *enc(): use hard_indices method instead of typical forward pass to avoid gumbel 
      """
      x = self.vae.hard_indices(image)
