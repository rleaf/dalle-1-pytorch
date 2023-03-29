import torch
from torch import nn, einsum
import torch.nn.functional as F

torch.manual_seed(10)

layer = nn.Embedding(2, 4)

# x = torch.randint(0, 2, (3, 5, 5, 2))
x = torch.randint(0, 2, (2, 3))
u = torch.rand(5)
normal = F.normalize(u, dim = 0)
out = layer(x)
# out2 = torch.matmul(x.permute(0, 3, 2, 1), layer.weight.to(dtype=torch.long)).permute(0, 3, 2, 1)
# print(x.shape)
# out21 = einsum('b n h w, n d -> b d h w', x, layer.weight.to(dtype=torch.long))
print('u', u)
print('normal', normal)
# print('layer', layer.weight)
# print('out2', out2.shape)
# print('out21', out21.shape)
