import torch
import torch.nn as nn

from transformer import Transformer
from dvae import dVAE


class DALLE(nn.Module):
   """
   INP  ->  enc(INP)=a  -> codebook(a)=b  ->
   preprocess(b)=c  ->  transformer(c)=d  ->
   (transformer(d)=d* x token_len)=d2  -> dec(d2)=OUT

   preprocess() {
      - BPE input text sequence
      - return concat text seq w/ codebook tokens
   }
   
   In training the Transformer autoregressively generates the next token 
   to represent the masked section of an image with the given text seq & codebook tokens
   provided as inp.
      
   Stage 1
      - Train dVAE on images where kl divergence in variational lower bound
        measures from auxiliary $q_phi$ to uniform categorical
   """
   def __init__(
      self,
      dim,
      # vae
      # img_size, ?
      tokens,
      codebook_dim,
      hidden_dim,
      channels,
      # transformer
      depth,
      head_dim,
      heads,
      ff_dim,
      dropout,
      ff_dropout
   ):
      super().__init__()

      self.dvae = dVAE(
         tokens = tokens,
         codebook_dim = codebook_dim,
         hidden_dim = hidden_dim,
         channels = channels
      )

      self.transformer = Transformer(
         dim = dim,
         depth = depth,
         head_dim = head_dim,
         heads = heads,
         ff_dim = ff_dim,
         dropout = dropout,
         ff_dropout = ff_dropout
      )

   
   def forward(self, x):
      pass