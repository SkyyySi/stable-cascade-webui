from __future__ import annotations

from typing import NoReturn, Optional, Literal

import torch

import gradio as gr

from diffusers import (
	AutoPipelineForImage2Image,
	StableCascadeDecoderPipeline,
	StableCascadePriorPipeline,
)

import cv2
import numpy as np
from PIL import Image

#from basicsr.archs.rrdbnet_arch import RRDBNet
#from realesrgan import RealESRGANer

prior   = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to("cuda")
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",     torch_dtype=torch.float16).to("cuda")

#upscaler_model = RealESRGANer(
#	scale=2,
#	model_path="/home/simon/Code/stable-diffusion-webui/models/RealESRGAN/RealESRGAN_x4plus.pth",
#	dni_weight=None,
#	model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
#	tile=0, # 4,
#	tile_pad=10,
#	pre_pad=0,
#	half=True,
#	gpu_id=0,
#)

def upscale(image: Image) -> Image:
	image_cv2 = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

	image_upscaled = upscaler_model.enhance(image_cv2, outscale=2)[0]

	image_upscaled_pillow = Image.fromarray(cv2.cvtColor(image_upscaled, cv2.COLOR_BGR2RGB))

	return image_upscaled_pillow

def latents_to_pil(latents):
	latents = (1 / 0.18215) * latents
	with torch.no_grad():
		image = pipeline_img2img.vae.decode(latents).sample
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
	images = (image * 255).round().astype("uint8")
	pil_images = [Image.fromarray(image) for image in images]
	return pil_images

def generate_image(
		prompt: str = "highres, masterpice, high quality, best quality, 4k, uhd",
		negative_prompt: str = "lowres, low quality, bad, ugly, glitch, distortion, mutant, disgust",
		width: int = 1024,
		height: int = 1024,
		steps: int = 20,
		cfg_scale: int = 10,
		progress = gr.Progress(),
):
	highres_steps: int = 10
	def callback(current_step: int, current_stage: Literal["first_pass", "second_pass"], timestamp, latents):
		if current_stage == "second_pass":
			current_step += highres_steps

		progress(progress=(current_step, steps + highres_steps))
		current_preview = latents_to_pil(latents)
		print(current_step, current_preview)
		yield from current_preview

	def interrupt_callback(pipe, i, t, callback_kwargs):
		#print(pipe, i, t, callback_kwargs)
		pass

	prior_output = prior(
		prompt=prompt,
		negative_prompt=negative_prompt,
		width=width,
		height=height,
		guidance_scale=cfg_scale,
		num_inference_steps=steps,
		num_images_per_prompt=1,
		callback_on_step_end=interrupt_callback,
	)

	decoder_output = decoder(
		image_embeddings=prior_output.image_embeddings.half(),
		prompt=prompt,
		negative_prompt=negative_prompt,
		guidance_scale=0.0,
		output_type="pil",
		num_inference_steps=10
	)

	final_image = decoder_output.images[0]

	return final_image

	#except Exception as e:
	
	#	print(f"ERROR: {e}")

	
	#	return None

def run_ui() -> NoReturn:
	demo = gr.Interface(
		fn=generate_image,
		inputs=[
			gr.Textbox("highres, masterpice, high quality, best quality, 4k, uhd", label="Prompt"),
			gr.Textbox("lowres, low quality, bad, ugly, glitch, distortion, mutant, disgust", label="Negative prompt"),
			gr.Slider(32, 2048, 1024, label="Width"),
			gr.Slider(32, 2048, 1024, label="Height"),
			gr.Slider(1, 100, 35, label="Inference steps"),
			gr.Slider(0, 20, 10, label="CFG scale"),
		],
		outputs=[gr.Image()],
	)

	demo.launch(
		server_name="0.0.0.0",
	)

def main() -> NoReturn:
	run_ui()

if __name__ == "__main__":
	main()
