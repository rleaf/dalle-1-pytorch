import torch
import torch.nn as nn

"""
Can standardly create the attn mechanism presented w/ what's below in the comments
however I'd like to try and do a more efficient operation where operations are done
in batches.

enc(a)=b  ->  codebook(b)=c  ->  transformer(c)=seq  ->  dec(seq)
"""

class Attention(nn.Module):
   """
   qkv(x)=x2  ->  preprocess(x2)=q,k,v  ->  sdp(q, k, v)=out

   preprocess() {
      - aliquot tensor x2 to segregate q, k, and v into a list and store in qkv
      - iterate through the list and isolate both head & head_dim in each element and
         the q, k, and v from each other
      - return q, k, v represented as 4d tensors composed (batch, head, inp_dim[1], head_dim)
   }
   """
   def __init__(self, dim, head_dim, heads):
      super().__init__()
      self.heads = heads
      self.inner_dim = head_dim * heads
      self.qkv = nn.Linear(dim, self.inner_dim * 3, bias = False) # (B, N, M) -> (B, N, D)

   def forward(self, x, mask = None): # (B, N, M)
      qkv = self.qkv(x)  # (B, N, D)
      qkv = qkv.chunk(3, dim = -1) # ((B, N, D/3) * 3)
      b, n, d = qkv[0].shape # (B, N, inner_dim)

      q, k, v = [qkv[i].reshape(b, n, self.heads, d // self.heads).permute(0, 2, 1, 3) \
          for i in range(len(qkv))] # (B, N, H, head_dim) -> (B, H, N, head_dim)
      attn = torch.matmul(q, k.permute(0, 1, 3, 2)) # (B, H, N, N)
      attn = attn / q.shape[-1] ** (1/2)

      if mask:
         pass

      attn_softmax = attn.softmax(dim = -1)
      out = torch.matmul(attn_softmax, v)
      return out 

torch.manual_seed(0)
x = torch.rand((2, 3, 4))
attn = Attention(4, 6, 8)
out = attn(x)
print('out', out.shape)

class Decoder(nn.Module):
   def __init__(self,
      num_heads,
      emb_dim,
      ff_dim,
      dropout,
      num_dec) -> None:
      super().__init__()
      """
      (vanilla dec)
      mask_attn(a)=b  ->  res_norm(b)=b2  ->  cross_attn(enc,b2)=c  ->
      res_norm(c)=c2  ->  ffn(c2)=d  ->  res_norm(d)=d2  ->  Linear(d2)=e  ->
      softmax(e)=out

      (sparse transformer: https://arxiv.org/pdf/1904.10509.pdf)
      this may take a bit (9/19/22)
      """

      def forward(self, x):
         pass

class Transformer(nn.Module):
   def __init__(self,
      num_heads,
      emb_dim,
      ff_dim,
      dropout,
      num_dec_layers) -> None:
      super().__init__()

      self.decoder = Decoder(num_heads, emb_dim, ff_dim, dropout, num_dec_layers)

      def forward(self, x):
         pass



# def sdp(q, k, v):
#    e = torch.bmm(q, k.transpose(-1, -2))
#    weights_softmax = (e / q.shape[-1] ** 0.5).softmax(dim = -1)
#    y = torch.bmm(weights_softmax, v)
#    return y #, weights_softmax

# class Attention(nn.Module):
#    def __init__(self, dim, inner_dim, heads) -> None:
#       super().__init__()

#       self.q = nn.Linear(dim, inner_dim)
#       self.k = nn.Linear(dim, inner_dim)
#       self.v = nn.Linear(dim, inner_dim)
#       self.linear = nn.Linear(inner_dim * heads, dim)
#       self.heads = heads
#       head_output = []

#       def forward(self, x):
#          for i in self.heads:
#             q = self.q(x)
#             k = self.k(x)
#             v = self.v(x)
#             y =  sdp(q, k, v)

#             head_output.append(y)

#          cat = head_output.cat(dim = -1)
#          return self.linear(cat)

# class Decoder(nn.Module):
#    def __init__(self, dim, inner_dim, heads, dropout, ff_dim) -> None:
#       super().__init__()

#       self.maskattn = Attention(dim, inner_dim, heads)
#       # self.attn2 = 
#       self.ff = nn.Sequential(
#          nn.Linear(dim, ff_dim),
#          nn.ReLU(),
#          nn.Linear(ff_dim, dim),
#          nn.LayerNorm(dim, eps = 1e-10),
#          nn.Dropout(dropout)
#       )

#       def forward(self, x):
#          pass


# class Transformer(nn.Module):
#    def __init__(self,
#       dim,
#       heads,
#       depth,
#       dropout,
#       ff_dim,
#       inner_dim) -> None:
#       super().__init__()
   
#       self.layers = nn.ModuleList(
#          [Decoder(dim, heads, dropout, ff_dim, inner_dim) for _ in range(depth)])

#       def forward(self, x):
#          pass


# model = Transformer(4, 6, 8, 0.1, 20, 30)
# x = torch.rand((20, 4))
# y = model(x)
# print(y.shape)
