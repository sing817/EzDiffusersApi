from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from datetime import datetime
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from io import BytesIO

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                            torch_dtype=torch.float16,
                                            safety_checker=None,
                                            requires_safety_checker=False,
                                            )
pipe.to("cuda")
# def image_grid(imgs, rows, cols):
#     assert len(imgs) == rows*cols

#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     grid_w, grid_h = grid.size
    
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i%cols*w, i//cols*h))
#     return grid

pipe.load_lora_weights("xbesing/chinese_ink_style2")
# pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()

# pipe.save_lora_weights("./lora_save/pytorch_lora_weights")

# pipe.unet.load_attn_procs("./output
# _tea/pytorch_lora_weights.bin")
# "liuyi:3, birds and nature:2, high aestheic, white background"
generator=torch.Generator(device="cuda").manual_seed(0)

# img_path ="backpack.jpg"

images = pipe("chinese ink painting,bird and nature,,green plants, white background",
             negative_prompt="low quality,low acsthetic,horrible",
             num_inference_steps=30,
             guidance_scale=7.5,
             height =2048, width=512,
             generator=generator,
            #  num_images_per_prompt=2
             ).images[0]

# images = pipe("high acsthetic,a backpack ,a VITA Lemon Tea , black background, studio lighting",
#              negative_prompt="low quality,low acsthetic,horrible",
#              image=img_path,
#              num_inference_steps=30,
#              guidance_scale=7.5,
#              strength=0.7
#             #  generator=generator,
#             #  num_images_per_prompt=2
#              ).images[0] 

# grid = image_grid(images, rows=1, cols=2)

now = datetime.now()
formatted_time = now.strftime("%d_%H%M%S")
filename = f"./testoutput/{formatted_time}test.png"
# grid.save(f"./testoutput/{formatted_time}_test.png")
images.save(filename)