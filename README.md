# Learning DALL-E
Understanding the DALL-E model by trying to build a *much* smaller one. <a href='https://arxiv.org/pdf/2102.12092.pdf'>Link</a> to paper.

Unfortunately all training will probably be cut short because I'm not comfortable with my 1070 FE sitting at 90c for long periods of time.

---

## Updates

#### 9/4/22
Partially trained dVAE on the MNIST dataset.

`10 epochs`, `128 batch_size`, `256 tokens`, `256 codebook_dim`, `128 hidden_dim`, `1e-3 lr`, `1 channel` (MNIST is b/w) 
   

<p align='center'>
400th iteration, 0th epoch
<img src='./dvae_generation/dvae_generationE0.jpg'>
400th iteration, 1st epoch
<img src='./dvae_generation/dvae_generationE1.jpg'>
0th iteration, 2nd epoch
<img src='./dvae_generation/dvae_generationE2.jpg'>
</p>