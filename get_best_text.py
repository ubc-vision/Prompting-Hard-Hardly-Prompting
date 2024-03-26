import open_clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)

from diffusers import StableDiffusionPipeline
from diffusers import PNDMScheduler

def measure_clip_similarity(orig_images, pred_images, clip_model, device):
    with torch.no_grad():
        orig_feat = clip_model.encode_image(orig_images)
        orig_feat = orig_feat / orig_feat.norm(dim=1, keepdim=True)

        pred_feat = clip_model.encode_image(pred_images)
        pred_feat = pred_feat / pred_feat.norm(dim=1, keepdim=True)
        return (orig_feat @ pred_feat.t()).mean().item()

model_id = "runwayml/stable-diffusion-v1-5"
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir='.')

weight_dtype = torch.float16

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=weight_dtype, cache_dir='.')
pipe = pipe.to(device)
image_length = 512

best_loss=0.
step =0
image_path = './ldm/data/image.png'
orig_image = Image.open(image_path).convert('RGB')
with open('./logs_forward_pass/prompt_file.txt','r') as textfile:
    prompt = textfile.readlines()
    for prompt_l in prompt:
        step=step+1
        if step % 1 == 0:
            with torch.no_grad():
                pred_imgs = pipe(
                    prompt_l,
                    num_images_per_prompt=1,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    height=image_length,
                    width=image_length,
                    ).images
                orig_images_temp = [clip_preprocess(orig_image).unsqueeze(0)]
                orig_images_t = torch.cat(orig_images_temp).to(device)
                pred_imgs_temp = [clip_preprocess(i).unsqueeze(0) for i in pred_imgs]
                pred_imgs_t = torch.cat(pred_imgs_temp).to(device)
                eval_loss = measure_clip_similarity(orig_images_t, pred_imgs_t, clip_model, device)
                print(step)

                if best_loss < eval_loss:
                    best_loss = eval_loss
                    best_text = prompt_l
                    best_pred = pred_imgs[0]
best_pred.save('./logs/pred_image.png')
print()
print(f"Best shot: consine similarity: {best_loss:.3f}")
print(f"text: {best_text}")
