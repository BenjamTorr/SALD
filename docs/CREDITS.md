# Credits

Parts of this codebase were adapted from the "Explaining AI" educational diffusion-model repos:

- [DDPM-Pytorch](https://github.com/explainingai-code/DDPM-Pytorch) — noise scheduler used in `ddpm.py`
- [StableDiffusion-PyTorch](https://github.com/explainingai-code/StableDiffusion-PyTorch) — VAE blocks in `vae/unet_vae.py`, conditional U-Net in `diffusion/unet_conditional_1d.py`
- [DiT-PyTorch](https://github.com/explainingai-code/DiT-PyTorch) — attention/patch-embedding blocks in `diffusion/dit_FiLM.py`

Code was adapted (not copied verbatim) for 1D sequences and the conditioning setup used here.
