<h1>MDAI Graphic Designer API</h1>
<p>version 1.31

歡迎使用 MDAI設計師 API！此 API 可以讓您根據自訂的文字提示或圖片風格變換生成高品質的圖片。

Welcome to the MDAI Graphic Designer API! This API allows you to generate high-quality images with customized text prompts or image style change.
<h2>API URL</h2>
The API can be accessed at `http://2dgpn.ai.marveldigital.com`
<h2>Chinese Support接受中文輸入</h2>
此 API 支援中文文字輸入。如果您的提示包含中文字元，API 會在生成圖片前自動將其翻譯成英文。然而，需要注意的是中文翻譯可能不正確，因此生成的圖片可能與輸入不相符。
<p>

The API supports Chinese text prompts. If your prompt contains Chinese characters, the API will automatically translate it to English before generating the image. However, it should be noted that the Chinese translate may be wrong and the generated images may not match the input if the Chinese translation is inaccurate.
<h2>Getting Started<h2>
<h3>功能介紹Endpoints<h3>
<blockquote>"/text2image/" 文字描述生成圖像</blockquote>
生成並返回基於提供的提示的圖片。如果提示包含中文字符，API 將先將其翻譯成英文。
<p>
Generates and returns an image based on the provided prompt. If the prompt includes Chinese characters, the API will translate it to English first.
<p>
<blockquote>"/getimage/" 文字描述生成圖像,直接返回可下載的附件而無需解碼</blockquote>
生成並返回基於提供的提示的圖片。如果提示包含中文字符，API 將先將其翻譯成英文。此外，生成的圖片將作為可下載的附件返回給用戶。
<p>Generates and returns an image based on the provided prompt. If the prompt includes Chinese characters, the API will translate it to English first. Additionally, the generated image will be returned as an attachment that can be downloaded by the user.
<p>
<blockquote>"/image2image/" 圖像轉變圖像,風格轉化</blockquote>
此 API 需要一個提示字符串和一個圖片文件作為輸入。API 將先將其翻譯成英文再執行圖像轉換。
<p>
It expects a prompt string and an image file as input.The prompt string is used to guide the image translation prompt or original prompt is then passed to the api,which performs the actual image translation.
<p>
<blockquote>"/imagepaint/" 變換指定目標</blockquote>
與圖像轉變圖像類似，但它可以輸入一個遮罩層來控制您需要更改的圖像區域。
<p>
Same like image to image,but it can input a mask layer to control your image where need to change

<blockquote>"/upscaler/" AI放大圖像 x4</blockquote>
以AI 放大圖像4倍並保留細節。
<p>
Uses AI to enlarge images by 4x and preserve details.
<h3>Usage 使用說明</h3>
<blockquote>"/text2image/{prompt}"
<p>Prompt: str</blockquote>
向 <b>http://2dgpn.ai.marveldigital.com/text2image/{prompt}</b> 發送 <b>GET</b> 請求，其中包含您要生成的<b>prompts</b>。API將以 <b>base64</b> 編碼的PNG圖像回應。
<p>
Send a <b>GET</b> request to <b>http://2dgpn.ai.marveldigital.com/text2image/{prompt}</b> with the <b>prompts</b> you want to generate an image from. The response will be the generated PNG image encoded in <b>base64</b>.
<p>
<p>
<p>
<blockquote>"/getimage/{prompt}"
<p>Prompt: str</blockquote>
向 <b>http://2dgpn.ai.marveldigital.com/getimage/{prompt}</b> 發送 <b>GET</b> 請求，其中包含您要生成圖片的<b>prompt</b>。ＡＰＩ將返回<b>PNG格式的圖片附件</b>作為回應。
<p>
Send a <b>GET</b> request to <b>http://2dgpn.ai.marveldigital.com/getimage/{prompt}</b> with the <b>prompts</b> you want to generate an image from. The response will be the generated image as an <b>attachment</b> that can be downloaded by the user.
<p>
<p>
<p>
<blockquote>"/image2image/{prompt}"
<p>Prompt: str
<p>img: file</blockquote>
向 <b>http://2dgpn.ai.marveldigital.com/image2image/</b> 發送 <b>POST</b> 請求，其中包含您要生成PNG圖片的<b>prompt</b>和您的<b>輸入圖片 "img"</b>。API將以 <b>base64</b> 編碼的PNG圖像應回應。
<p>
Send a <b>POST</b> request to <b>http://2dgpn.ai.marveldigital.com/image2image/</b> with the <b>prompts</b> and your <b>input image "img"</b> you want to generate an image from. The response will be the generated PNG image encoded in <b>base64</b>.
<p>
<blockquote>"/imagepaint/{prompt}"
<p>Prompt: str
<p>img: file
<p>Maskimg: file</blockquote>
<p>
向 <b>http://2dgpn.ai.marveldigital.com/image2image/</b> 發送 <b>POST</b> 請求，其中包含<b>prompt</b>，<b>輸入圖片 "img"</b> 和 <b>遮罩圖片 "Maskimg"</b>，將以 <b>base64</b> 編碼的PNG圖像回應。
<p>
Send a <b>POST</b> request to <b>http://2dgpn.ai.marveldigital.com/image2image/</b> with the <b>prompt</b> , <b>input image "img"</b> and <b>mask image "Maskimg"</b> ,it will generate an PNG image encoded in <b>base64</b>.
<blockquote>"/upscalerX4/"
<p>img: file
</blockquote>
<p>
向 <b>http://2dgpn.ai.marveldigital.com/upscalerX4/</b> 發送 <b>POST</b> 請求，其中包含<b>輸入圖片 "img"</b>，將以放大了4倍的 <b>base64</b> 編碼PNG圖像作回應。
<p>
Send a <b>POST</b> request to <b>http://2dgpn.ai.marveldigital.com/upscalerX4/</b> with <b>input image "img"</b>,it will up scale an PNG image encoded in <b>base64</b> by 4.
<h1>Example使用例子</h1>
以下是一些使用API的例子供參考

<h3>"/text2image/"</h3>

```Python

import requests
from PIL import Image
import io

import base64
# Supports Chinese text prompts可以輸入中文字
prompt = "一名太空人在太空中飛行,科幻,水彩風格"

response = requests.get(f"http://2dgpn.ai.marveldigital.com/text2image/{prompt}")
img = Image.open(io.BytesIO(base64.b64decode(response.content)))
img.show()

```
<h3>"/getimage/"</h3>

```Python
import requests

prompt = "An astronaut flying in space, science fiction, water color style"

response = requests.get(f"http://2dgpn.ai.marveldigital.com/getimage/{prompt}")
with open('generated_image.png', 'wb') as f:
    f.write(response.content)
```
<h3>"/image2image/"</h3>

```Python
import requests
from io import BytesIO
from PIL import Image 
import base64 

API_image2iamge = "http://2dgpn.ai.marveldigital.com/image2image/"

img_path = "./harbour-thumbnail.jpg"
prompt = "Makoto Shinkai style"

with open(img_path,"rb") as f:
    img_data = f.read()

response = requests.post(
    API_image2iamge+"?prompt="+prompt,
    files = {"img": img_data})

output_img = Image.open(BytesIO(base64.b64decode(response.content)))

output_img.show()

```
<h3>"/imagepaint/"</h3>

```Python
import requests
from io import BytesIO
from PIL import Image 
import base64

API_inpaint = "http://2dgpn.ai.marveldigital.com/imagepaint/"
img_path = "./harbour-thumbnail.jpg"
mask_path = "./harbour-thumbnail.jpg"
prompt = "Makoto Shinkai style"

with open(img_path,"rb") as f:
    img_data = f.read()
with open(mask_path,"rb") as f:
    mask_data = f.read()


response = requests.post(
    API_inpaint + "?prompt=" + prompt,
    files = {"img": ("input_image.jpg",img_data),"Maskimg":("mask_layer_image.jpg", mask_data)})

if response.ok:
    output_img = Image.open(BytesIO(base64.b64decode(response.content)))
    output_img.show()
    print("Inpainting comleted successfully!")
else:
    print("Error:", response.status_code, response.reason)

```

<h3>/upscalerX4/</h3>

```Python
import requests
from PIL import Image
import io

url = "http://2dgpn.ai.marveldigital.com/upscalerX4/"
image_path = "your_input_image.png"

# Open the image and convert it to bytes
image = Image.open(image_path)
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG')
img_bytes = img_byte_arr.getvalue()

# Send the POST request with the image data
response = requests.post(url, files={"img": img_bytes})

# Get the base64 encoded response image
response_image_base64 = response.text

# Decode the base64 encoded image and display it
response_image_bytes = base64.b64decode(response_image_base64)
response_image = Image.open(io.BytesIO(response_image_bytes))
response_image.show()

```


24/03/2023</i>
