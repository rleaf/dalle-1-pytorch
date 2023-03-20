import torch
import torch.nn as nn

def mask(x):
   # Fix dimensions
   mask = torch.triu(
      torch.full((x.shape[2], x.shape[2]), float('-inf'), dtype=torch.bool),
      diagonal=1
   )
   mask = mask.repeat((x.shape[0], x.shape[1], 1, 1))
   return mask

class Attention(nn.Module):
   """
   Giving it a go in batches

   qkv(x)=x2  ->  preprocess(x2)=q,k,v  ->
   sdp(q, k, v)=out  ->  self.out(out)=out

   preprocess() {
      - aliquot tensor x2 to segregate q, k, and v into a list and store in qkv
      - iterate through the list and isolate both head & head_dim in each element and
         the q, k, and v from each other
      - return q, k, v represented as 4d tensors composed (batch, inp_dim[1], head, head_dim)
   }
   """
   def __init__(self, dim, head_dim, heads, dropout):
      super().__init__()

      inner_dim = head_dim * heads
      self.heads = heads
      self.qkv = nn.Linear(dim, inner_dim * 3, bias = False) # (B, N, M) -> (B, N, D)
      self.out = nn.Sequential(
         nn.Linear(inner_dim, dim),
         nn.Dropout(dropout)
      )

   def forward(self, x, mask = None): # (B, N, M)
      qkv = self.qkv(x)  # (B, N, D)
      qkv = qkv.chunk(3, dim = -1) # ((B, N, D/3) * 3)
      b, n, d = qkv[0].shape # (B, N, inner_dim)

      q, k, v = [qkv[i].reshape(b, self.heads, d // self.heads, n) \
         for i in range(len(qkv))]  # (B, H * head_dim, N) -> (B, H, head_dim, N)

      attn = torch.matmul(q, k.permute(0, 1, 3, 2)) # (B, H, head_dim, head_dim)
      attn = attn / q.shape[-1] ** (1/2)

      if mask is not None:
         attn = attn.masked_fill_(mask, -1e9)

      attn_softmax = attn.softmax(dim=-1)
      
      out = torch.matmul(attn_softmax, v) # (B, H, N, head_dim)
      out = out.reshape(b, n, -1) # (B, N, H * head_dim)
      out = self.out(out) # (B, N, inner_dim) -> (B, N, M)
      return out

# torch.manual_seed(0)
# attn = Attention(4, 6, 8, 0.0)
# x = torch.rand((2, 3, 4))
# m = mask(x)
# out = attn(x)
# print('out', out.shape)

class FeedForward(nn.Module):
   def __init__(self, dim, ff_dim, dropout):
      super().__init__()

      self.mlp = nn.Sequential(
         nn.Linear(dim, ff_dim),
         nn.ReLU(),
         nn.Dropout(dropout),
         nn.Linear(ff_dim, dim),
         nn.LayerNorm(dim)
      )
   
   def forward(self, x):
      return self.mlp(x)

class Transformer(nn.Module):
   """
   (vanilla dec)
   mask_attn(a)=b  ->  res_norm(b)=b2  ->  cross_attn(enc,b2)=c  ->
   res_norm(c)=c2  ->  ffn(c2)=d  ->  res_norm(d)=d2  ->  Linear(d2)=e  ->
   softmax(e)=out

   (sparse transformer: https://arxiv.org/pdf/1904.10509.pdf)
   this may take a bit (9/19/22)
   """
   def __init__(self,
      dim,
      depth,
      head_dim,
      heads,
      ff_dim,
      dropout = 0.0,
      ff_dropout = 0.0):
      super().__init__()

      self.layers = nn.ModuleList([])
      self.norm = nn.LayerNorm(dim)

      for _ in range(depth):
         self.layers.append(nn.ModuleList([
            Attention(dim = dim, head_dim = head_dim, heads = heads, dropout = dropout),
            FeedForward(dim = dim, ff_dim = ff_dim, dropout = ff_dropout)
         ]))

   def forward(self, x):
      """
      enc(a) = b -> codebook(b) = c -> transformer(c) = seq -> dec(seq)
      """
      for attn, ff in self.layers:
         x = attn(x) + x
         # x = self.norm(attn(x) + x)
         x = ff(x)

      return x


torch.manual_seed(0)
model = Transformer(4, 3, 6, 8, 4)
x = torch.rand((2, 3, 4))
out = model(x)
print('out', out.shape)
