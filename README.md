# Prompting Hard or Hardly Prompting: Prompt Inversion for Text-to-Image Diffusion Models


<p align="center">
  <img width="450" height="350" src="./assets/teaser.png" hspace="30">
</p>

This repository is the PyTorch implementation of the paper:

[**Prompting Hard or Hardly Prompting: Prompt Inversion for Text-to-Image Diffusion Models (CVPR 2024)**](https://arxiv.org/abs/2312.12416)

[Shweta Mahajan](https://s-mahajan.github.io/), [Tanzila Rahman](https://sites.google.com/view/tanzila-rahman/home), [Kwang Moo Yi](https://www.cs.ubc.ca/~kmyi/), [Leonid Sigal](https://www.cs.ubc.ca/~lsigal/)


## Requirements
The following code is based on the [Stable-diffusion-repository](https://github.com/CompVis/stable-diffusion).


## Navigating the Prompt Inversion
The PH2P code modifies the following modules:
1. ddpm.py with the LBFGS optimizer and saving the prompts after each iteration during optimization.
2. Patch the ```Clip_transformer``` to ```localclip_transformer```. ```LocalCustomTokenEmbedding``` with the projection algorithm.
3. Specify the model path in ```main_textual_inversion.py```.
4. [embedding_matrix.pt](https://drive.google.com/file/d/1zzTZUsNBilHpi-1rEOoaj6fUqZ0sUCeE/view?usp=drive_link) contains the embeddings for the CLIP vocabulary (vocab.json)
5. Download the model checkpoint (stable diffusion v1.4 or v1.5) and save in ```models/ldm/stable-diffusion-v1/model.ckpt```


For running the prompt inversion specify image path in inversion_config.json
 ```
		python main_textual_inversion.py
 ```
The prompts will be saved in ```./logs_forward_pass/```.

The best prompt for a given image is obtained from the maximum clip similarity between the target image and the generated image for a prompt.
This additionally requires ```transformers 4.25.1``` and ```diffusers 0.12.1```
 
 ```
		python get_best_text.py
 ```

## Bibtex

	@inproceedings{ph2p2024cvpr,
	  title     = {Prompting Hard or Hardly Prompting: Prompt Inversion for Text-to-Image Diffusion Models},
	  author    = {Shweta Mahajan, Tanzila Rahman, Kwang Moo Yi, Leonid Sigal},
	  booktitle = {CVPR 2024 (To appear)},
	  year = {2024}
	}
