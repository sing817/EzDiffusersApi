# by SingWan,MDAI

from auth_token import auth_token
from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from io import BytesIO
import base64
import re
from datetime import datetime
from PIL import Image
import subprocess
import cv2
import numpy as np


description = """
歡迎使用 MDAI設計師 API！此 API 可以讓您根據自訂的文字提示生成高品質的圖片或圖片風格變換。

Welcome to the MDAI Graphic Designer API! This API allows you to generate high-quality images with customized text prompts or image style change.

"""
app = FastAPI(title="MDAI Graphic Designer API",
              description=description,
              contact={
                  "name": "Wan Wai Sing(SingWan)",
                  "email": "0wwsing0@gmail.com"
              },
              version="1.52",
              docs_url="/api_docs")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=origins,
    allow_methods=["*"],
)

device = "cuda"

##translate model##

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translationmodel = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

##translate##
def translate_text(prompt):
    if re.search('[\u4e00-\u9fff]', prompt):
        translated_prompt = tokenizer(prompt, return_tensors="pt")
        translated_output = translationmodel.generate(**translated_prompt)
        translated_prompt = tokenizer.decode(translated_output[0],
                                             skip_special_tokens=True)
        return translated_prompt
    else:
        return prompt

##image process

#skin smoothing
def skin_smoothing(image,faces,kernel_size=5, sigma=20):
    image = image.astype(np.float32)

    for(x,y,w,h) in faces:

        face = image[y:y+h, x:x+w]

        blurred_face = cv2.GaussianBlur(face,(kernel_size, kernel_size),sigma)

        image[y:y+h, x:x+w] = blurred_face

    smoothed = image.astype(np.uint8)

    return smoothed
##stable-diffusion-2-1##

sdmodel_2 = "stabilityai/stable-diffusion-2-1"
sdmodel_depth = "stabilityai/stable-diffusion-2-depth"
sdmodel_inpaint = "stabilityai/stable-diffusion-2-inpainting"
sdmodel_upscaler = "stabilityai/stable-diffusion-x4-upscaler"
modelR='SG161222/Realistic_Vision_V1.4'
# modelAnime="gsdf/Counterfeit-V3.0"

scheduler = EulerAncestralDiscreteScheduler()
sdpipe = StableDiffusionPipeline.from_pretrained(sdmodel_2,safety_checker=None,
                                             requires_safety_checker=False,
                                             torch_dtype=torch.float16,
                                            ).to(device)

# xformers 
sdpipe.enable_xformers_memory_efficient_attention()

# enable attention slicing can use less vram
sdpipe.enable_attention_slicing()

# ##stable_diffusion-2-image2image##

image2imagepipe=StableDiffusionImg2ImgPipeline(**sdpipe.components).to(device)

image2imagepipe.enable_attention_slicing(slice_size="max")

image2imagepipe.enable_xformers_memory_efficient_attention()

# image2imagepipe.unet.to(memory_format=torch.channels_last)

### ink lora style
image2imagepipeink=StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                  scheduler = scheduler,
                                                                  ).to(device)

image2imagepipeink.load_lora_weights("xbesing/chinese_ink_style2")

image2imagepipeink.enable_xformers_memory_efficient_attention()

image2imagepipesketch=StableDiffusionImg2ImgPipeline.from_pretrained("./sketch2/",
                                                    safety_checker=None,
                                                    requires_safety_checker=False,
                                                    torch_dtype=torch.float16,
                                                     ).to(device)

image2imagepipesketch.enable_xformers_memory_efficient_attention()

image2imagepipesketch.enable_attention_slicing(slice_size="max")

# ##stable_diffusion-2-inpaint##

inpaintingpipe=StableDiffusionInpaintPipeline(**sdpipe.components)
inpaintingpipe=inpaintingpipe.to(device)

# inpaintingpipe.enable_attention_slicing()

inpaintingpipe.enable_xformers_memory_efficient_attention()

# inpaintingpipe.unet.to(memory_format=torch.channels_last)

torch.set_grad_enabled(False)

# #stable-diffusion-x4-upscaler##

# upscaler = StableDiffusionUpscalePipeline.from_pretrained(
#     sdmodel_upscaler, torch_dtype=torch.float16)
# upscaler = upscaler.to(device)

# upscaler.enable_xformers_memory_efficient_attention()

# API global setttings

## public prompts

## public negative prompts

negative_prompt = "horrible, bad anatomy, ugly, disgusting, amputation, low quality:1.4, low aesthetic:1.4,word,text,signature,sore"

# API: test API is working.

@app.get("/{prompt}", tags=["testing"])
async def servertest(prompt: str):
    return {"message": "server Connected!"}

# API: Get text and Reture base64 image.

@app.get("/text2image/{prompt}", tags=["text to image"])
def text2image(prompt: str):
    # with autocast(device):
    translate_prompt = translate_text(prompt)

    image = sdpipe(
        translate_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7,
        num_inference_steps=50,
        # num_images_per_prompt=1
        ).images[0]
 
    # up_image = upscaler(translate_prompt, image = image,
    #                 num_inference_steps=50
    #                 ).images[0]

    newfilename = 'text2img' + datetime.now().strftime("%m%d_%H%M") + ".png"
    image_path = "./images/output/" + newfilename
    image.save(image_path)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    # image = None
    # buffer = None
    return Response(content=imgstr, media_type="image/png")

# API: Get text and Return file attachment.

@app.get("/getimage/{prompt}", tags=["text to image"])
def text2imagewithattchment(prompt: str):
    # with autocast(device):
    translate_prompt = translate_text(prompt)

    image = sdpipe(translate_prompt,
                 negative_prompt=negative_prompt,
                 guidance_scale=7.5,
                 height=512,
                 width=512,
                 num_inference_steps=10
                 #  num_images_per_prompt=1
                 ).images[0]
    
    # low_res_img = image.resize((256,256))
    # up_image = upscaler(translate_prompt, image = low_res_img,
    #                 num_inference_steps=10
    #                 ).images[0]

    newfilename = 'getimage' + datetime.now().strftime("%m%d_%H%M") + ".png"
    image_path = "./images/output/" + newfilename
    image.save(image_path)

    return FileResponse(image_path,
                        media_type='image/png',
                        headers={"Content-Disposition": "attachment"})

# API: image2image

@app.post("/image2image/", tags=["image to image"])
async def image2image(prompt: str, img: UploadFile = File(...)):

    with open(f"./images/input/{img.filename}", "wb") as buffer:
        buffer.write(await img.read())
    inputimage = Image.open(f"./images/input/{img.filename}")
    if inputimage.mode == 'RGBA':
        inputimage = inputimage.convert('RGB')
    inputimage = inputimage.resize((512, 512))

    translate_prompt = translate_text(prompt)
    print(translate_prompt + datetime.now().strftime("%d_%H%M%S"))
    
    
    
    outputimage = image2imagepipe(prompt=translate_prompt,
                                  image=inputimage,
                                  strength=0.40,
                                  negative_prompt=negative_prompt,
                                  num_inference_steps=80,
                                  guidance_scale=7.5,
                                  ).images[0]

    newfilename = 'img2img_' + datetime.now().strftime("%m%d_%H%M%S") + ".png"
    image_path = "./images/output/" + newfilename
    outputimage.save(image_path)
    buffer = BytesIO()
    outputimage.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    # outputimage = None
    # buffer = None

    return Response(content=imgstr, media_type="image/png")


@app.post("/3style/", tags=["image to image"])
async def style(img: UploadFile = File(...), choice:str = Form(...)):
    
    prompt_map = {
        "oil painting":'high aesthetic, (oil painting:2),Hand Painted,masterpiece',
        "ink painting":'high aesthatic,(chinese ink painting), high quality',
        "quick sketch":'(Hand Painted on paper), (pencil sketching:2), complete'
    }
    
    # inputimage = process_image(img)
    with open(f"./images/input/{img.filename}", "wb") as buffer:
        buffer.write(await img.read())
    inputimage = Image.open(f"./images/input/{img.filename}")
    if inputimage.mode == 'RGBA':
        inputimage = inputimage.convert('RGB')
    inputimage = inputimage.resize((512, 512))
    
    if choice not in prompt_map:
        return {"error": "Invalid choice.You can choice:'oil painting' or 'ink painting' or 'quick sketch'. "}
    
    prompt = prompt_map[choice]
    if choice=="ink painting":
        print("ink model load")
        outputimage = image2imagepipeink(prompt,
                            image=inputimage,
                            strength=0.3,
                            negative_prompt=negative_prompt,
                            num_inference_steps=30,
                            guidance_scale=7.5).images[0]
        
    elif choice=="quick sketch":
        print("sketch model load")
        outputimage = image2imagepipesketch(prompt,
                            image=inputimage,
                            strength=0.3,
                            negative_prompt=negative_prompt,
                            num_inference_steps=30,
                            guidance_scale=6.5).images[0]
        
    else:
        print("oil painting")
        outputimage = image2imagepipe(prompt,
                                    image=inputimage,
                                    strength=0.40,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=30,
                                    guidance_scale=7.5).images[0]

    newfilename = f'img2img2_' + datetime.now().strftime(
        "%m%d_%H%M%S")+".png"
    image_path = "./images/output/" + newfilename
    outputimage.save(image_path)
    buffer = BytesIO()
    outputimage.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())
    
    return Response(content=imgstr, media_type="image/png")


# API: inpaint

@app.post("/imagepaint/", tags=["image to image"])
async def inpaint(prompt: str,
                  img: UploadFile = File(...),
                  Maskimg: UploadFile = File(...)):
    # with autocast(device):

    # Load input image from memory
    inputimage = Image.open(BytesIO(await img.read()))
    inputimage = inputimage.resize((512, 512))

    # Load mask image from memory
    Maskimage = Image.open(BytesIO(await Maskimg.read()))
    Maskimage = Maskimage.resize((512, 512))

    translate_prompt = translate_text(prompt)

    outputimage = inpaintingpipe(translate_prompt,
                                 image=inputimage,
                                 mask_image=Maskimage,
                                 guidance_scale=7.5,
                                 num_inference_steps=50).images[0]

    newfilename = 'imgpaint' + datetime.now().strftime("%m%d_%H%M") + ".png"
    image_path = "./images/output/" + newfilename
    outputimage.save(image_path)
    buffer = BytesIO()
    outputimage.save(buffer, format="PNG", optimize=True)
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")


@app.post("/upscalerX4/", tags=["image proessing"])
async def upscaler(img: UploadFile = File(...)):
    #load input image
    inputimagepath = "./images/input/upscaler.png"
    inputimage = Image.open(BytesIO(await img.read()))
    # scale_image
    inputimage.save(inputimagepath)

    result = subprocess.run([
        "python", "./Real-ESRGAN/inference_realesrgan.py", "--model_path",
        "./Real-ESRGAN/weights/RealESRGAN_x4plus.pth", "-n",
        "RealESRGAN_x4plus", "-i", inputimagepath, "--output",
        "./images/output/"
    ],
                            capture_output=True,
                            text=True)
    print(result.stdout)

    result_image = Image.open("./images/output/upscaler_out.png")
    buffer = BytesIO()
    result_image.save(buffer, format="JPG", optimize=True, quality=80)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.post("/skinsmooth/", tags=["image proessing"])
async def skinsmooth(img:UploadFile = File(...)):

    input_image = Image.open(BytesIO(await img.read()))  
    input_image = np.array(input_image)

    face_casade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    faces = face_casade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5, minSize=(30,30))

    smoothed_image = skin_smoothing(input_image, faces)
    image_without_boxes = smoothed_image.copy()
    image_with_boxes = smoothed_image.copy()

    for (x,y,w,h) in faces:
        cv2.rectangle(image_with_boxes, (x,y), (x+w,y+h), (0,255,0), 2)

    newfilename_with_box = 'skin_debug' + datetime.now().strftime("%m%d_%H%M") + ".png"
    newfilename_without_box = 'skin' + datetime.now().strftime("%m%d_%H%M") + ".png"
    image_path = "./images/output/"

    pil_image_with_box = Image.fromarray(image_with_boxes)
    pil_image_without_box = Image.fromarray(image_without_boxes)

    pil_image_with_box.save(image_path + newfilename_with_box)
    pil_image_without_box.save(image_path + newfilename_without_box)

    buffer = BytesIO()
    pil_image_without_box.save(buffer, format="PNG", optimize=True)
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")

# by SingWan,MDAI

