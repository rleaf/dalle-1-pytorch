import torch
import torch.nn as nn


def sdp(q, k, v):
   e = torch.bmm(q, k.transpose(-1, -2))
   weights_softmax = (e / q.shape[-1] ** 0.5).softmax(dim = -1)
   y = torch.bmm(weights_softmax, v)
   return y #, weights_softmax

class Attention(nn.Module):
   def __init__(self, dim, inner_dim, heads) -> None:
      super().__init__()

      self.q = nn.Linear(dim, inner_dim)
      self.k = nn.Linear(dim, inner_dim)
      self.v = nn.Linear(dim, inner_dim)
      self.linear = nn.Linear(inner_dim * heads, dim)
      self.heads = heads
      head_output = []

      def forward(self, x):
         for i in self.heads:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            y =  sdp(q, k, v)

            head_output.append(y)

         cat = head_output.cat(dim = -1)
         return self.linear(cat)

class Decoder(nn.Module):
   def __init__(self, dim, inner_dim, heads, dropout, ff_dim) -> None:
      super().__init__()

      self.maskattn = Attention(dim, inner_dim, heads)
      # self.attn2 = 
      self.ff = nn.Sequential(
         nn.Linear(dim, ff_dim),
         nn.ReLU(),
         nn.Linear(ff_dim, dim),
         nn.LayerNorm(dim, eps = 1e-10),
         nn.Dropout(dropout)
      )

      def forward(self, x):
         pass


class Transformer(nn.Module):
   def __init__(self,
      dim,
      heads,
      depth,
      dropout,
      ff_dim,
      inner_dim) -> None:
      super().__init__()
   
      self.layers = nn.ModuleList(
         [Decoder(dim, heads, dropout, ff_dim, inner_dim) for _ in range(depth)])

      def forward(self, x):
         pass


model = Transformer(4, 6, 8, 0.1, 20, 30)
x = torch.rand((20, 4))
y = model(x)
print(y.shape)