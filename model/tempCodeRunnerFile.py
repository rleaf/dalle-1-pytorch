
import torch
model = Transformer(4, 6, 8, 0.1, 20, 30)
x = torch.rand((20, 4))
print(y.shape)
y = model(x)