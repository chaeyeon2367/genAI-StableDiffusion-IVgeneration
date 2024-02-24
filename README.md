
</br>

# 🌈AI Image and Video Generation Project with Stable Diffusion

</br> 

## 💡 Project Overview
This repository showcases the results of an AI image and video generation project using Stable Diffusion. The project involves utilizing the Stable Diffusion WebUI for Prompt Engineering, ControlNet, Dreambooth, and LoRA Generative AI models. Dreambooth and LoRA models have been trained on a custom dataset,and the generated content is included in this repository.

</br> 

## 🏷 Table of Contents

1. [Stable Diffusion Introduction and WebUI Installation](#1-stable-diffusion-introduction-and-webui-installation)
2. [Text to Image](#2-Text-to-Image) 
3. [Image to Image](#3-Image-to-Image)
4. [How to Write Prompts](#4-How-to-Write-Prompts)
5. [ControlNet Variants](#5-controlnet-variants)
6. [Dreambooth,LoRA Models Training](#6-Dreambooth-LoRA-Models-Training)
7. [Video Generation with Deforum](#7-video-generation-with-deforum)
8. [Animating Real-Person Videos with Move to Move](#8-animating-real-human-videos-with-move-to-move)
9. [Video Generation with Animatediff](#9-video-generation-with-animatediff)

</br> 

## 1. Stable Diffusion Introduction and WebUI Installation
</br> 

### 📌 What is Stable diffusion?

Stable Diffusion is a generative AI technique that involves the controlled diffusion of information throughout a system. Diffusion models are probabilistic models that describe how data, in this case, an image, changes or diffuses over time. The stable diffusion approach aims to create high-quality and diverse images by iteratively applying controlled diffusion processes to an initial image.

In the context of AI and creative applications, stable diffusion is often used to generate visually appealing and novel artworks. By manipulating the diffusion process through prompts and input parameters, users can guide the AI in creating unique and imaginative images. The stability in diffusion refers to the controlled and coherent evolution of the image during the generation process.

The technique is versatile, allowing users to explore a wide range of creative possibilities by influencing factors such as lighting, style, environment, and more. It is particularly popular in the field of generative art, where artists and AI enthusiasts leverage stable diffusion to produce captivating and diverse visual content.

</br>   

### 💾 Install Stable Diffusion Webui on Colab and Locally
</br> 

<img width="720" alt="Screenshot 2024-02-23 at 19 50 13" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/7b60a464-550e-4d94-9115-bdf96f4e0b68">

#### (1) Install stable-diffusion-webui on Mac
  - hardware requirements
    - Processor (CPU): Apple Silicon (M1 or M2) – Recommended CPUs include M1, M1 Pro, M1 Max, M2, M2 Pro, and M2 Max. Both efficient and performance cores are important.
    - Memory (RAM): Ideally, your machine should have 16 GB of memory or more.
    - Performance Comparison: Stable Diffusion runs slower on Mac. A similarly priced Windows PC with a dedicated GPU is expected to deliver images faster.
  - system requirements
    - You should have an Apple Silicon M1 or M2, with at least 8GB RAM. Your MacOS version should be at least 12.3. Click the Apple icon on the top left and click About this Mac. Update your MacOS before if necessary.   

  - Install [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) on Mac

    - Creating an Anaconda Virtual Environment
      
        `conda create --n YourEnvName`
    
        `conda activate YourEnvName`

       A new folder stable-diffusion-webui should be created under your home directory.

        `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
      
        `cd ~/stable-diffusion-webui;./webui.sh --no-half`
      
      
      <img width="512" alt="Screenshot 2024-02-23 at 13 51 26" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/3f8be52e-60aa-453d-93cb-876a3c425678">
      
      
      Open a web browser and click the following URL to start Stable Diffusion.

      `http://127.0.0.1:7860/`

 
      
      

#### (2) Install stable-diffusion-webui on Colab

  - Version 1. [AUTOMATIC1111](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)
  - Version 2. [digiclau](https://colab.research.google.com/github/DigiClau/stablediffusion_webui/blob/main/StableDiffusionWebUI_digiclau.ipynb) korean ver.

</br> 


### 🤖 Stable Diffusion models(checkpoint)

The Stable Diffusion model, akin to the artist who draws in the space provided by the Stable Diffusion webui, is the entity responsible for creating images. In other words, choosing a model is comparable to selecting the artist who will be drawing. While techniques like LoRA, embedding, hypernetwork, and others are capable of generating images, the model or checkpoint serves as the artist; without it, there is no one to create the artwork. Therefore, having the right model (checkpoint) is essential for the generation of images in the Stable Diffusion framework.

In the same way, choosing a model is akin to selecting the artist, and just as the style of an artwork varies depending on who is drawing, the images generated differ significantly based on the choice of the model. There are two main websites where you can download these models.

  - Civitai : <https://civitai.com>

  - Hugging face : <https://huggingface.co>

</br> 

#### 📂 Checkpoint(model) files (.safetensors , .ckpt)

The model files with the extensions .safetensors and .ckpt are related to the Stable Diffusion webui and represent different aspects of the model:

   - safetensors : This file contains tensors (data structures representing multi-dimensional arrays) related to the model.
It may include information about the model's architecture, parameters, and other essential components.
The ".safetensors" extension suggests that the data stored in this file is considered safe or stable for the model's functioning.

   - 'ckpt' : This file typically represents a checkpoint file and contains the saved weights and biases of the model.
It allows the model to be saved and restored at a later time, enabling users to continue training or deploy the model without starting from scratch.
The ".ckpt" extension is a common convention in machine learning to denote checkpoint files.

</br> 

## 2. Text to Image

</br>

### 🖥  Generated images using text-to-image in the webui

</br>

<img width="720" alt="Screenshot 2024-02-23 at 21 06 20" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/fae3e798-2e16-4a2f-8ac0-0204b72e2dda">

</br>



  - **Parameters**
    
    - **Sampling Steps** : the number of steps or iterations during the sampling process. A higher value may result in more detailed and refined images but will require more computational resources.
    - **Sampling Method** : Refers to the technique used for sampling during image generation. Different methods can influence the diversity and quality of generated images. 
    - **CFG Scale** : Stands for "config scale." It represents a scaling factor for the configuration settings, influencing the overall size and structure of the generated images.
    - **Batch Size** : The number of samples processed in one iteration. A larger batch size can speed up training but requires more memory.
    - **Size (Width, Height)** : Defines the dimensions of the generated images. Specified as width and height, this setting determines the resolution and aspect ratio of the output images.
    - **Seed** : A seed is an initial value used to start the randomization process. Setting a seed allows for reproducibility, ensuring that running the model with the same seed produces the same results.                          

</br>



### 📎 Images of cosmetic models generated with txt2img

Using Stable Diffusion for generating cosmetic model images through txt2img is an innovative and practical approach. It reflects the potential of AI-generated images for advertising and marketing purposes without relying on actual models. 

</br>

![Collage](https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/f9dbff8a-d12d-4649-9ff8-72ff512b1919)

  - Sampling Steps: 30
  - Sampling method: DPM++ 2M Karras
  - CFG scale: 4
  - Size: 512x720
  - Model: Realistic_Vision_V5.1_fp16-no-ema
  - VAE: vae-ft-ema-560000-ema-pruned.ckpt

```
Prompt : (realistic, photo-realistic:1.37), professional lighting, photon mapping, radiosity, 1girl, smile,
(holding a perfume:1.3),perfume, (medium shot), (looking at viewer:1), high quality, highres,
8k, accurate color reproduction, realistic texture,((simple background, white background)),
((wearing turtleneck sweater)), (extra deatailed), (best quality), 1girl, ((extra deatailed body:1.3)),
(realistic), narrow waist, (straight hair, medium-length hair, black hair, partedhairs:1.45), breasts,
pale skin, (realistic glistening skin:1.2), skindentation, masterpiece, (proportional eyes, same size eyes),
<lora:jwy___v1:1>

Negative prompt: 7dirtywords, easynegative, (nudity:1.3), nsfw, (worst quality:2.0), bad-artist, bad-hands-5
```

</br>


## 3. Image to Image

In Stable Diffusion, Img2img allows the generation of new images from existing ones. This concept is highly versatile, enabling the creation of images in various styles. Whether transforming realistic images into animated styles or generating images in different artistic expressions, Img2img provides a powerful tool for diverse and creative image synthesis. This feature is particularly advantageous for artists, designers, and content creators seeking flexibility and creative freedom in their image generation process.

</br>

### 📍 Image to Image generation

  I used the realisticVisionV60B1, revAnimized models for the generation of various styles of images.

  
</br>

- **Realistic image to animated image**

  ![Collage-5](https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/c3b04c72-cd16-4015-b441-ffe8aa043f9a)

</br>

- **Animated image to realistic image**


  ![Collage-6](https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/b1c7c07a-b649-4271-9add-e01d932501ed)

</br>

### 🎨 Utilize the Inpaint feature

</br>

- **Results with the Inpaint feature applied**
<img width="760" alt="Screenshot 2024-02-23 at 22 59 39" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/d3034705-ece9-4586-8e1f-1adc9fe16919">


</br>
</br>

In Stable Diffusion WebUI, the Inpaint feature is a powerful tool that allows the transformation of images using the Img2img model. Specifically, it can be employed to seamlessly replace or fill in missing parts of an image, enhancing its visual appeal. This functionality becomes particularly useful in scenarios where image editing or manipulation is required, such as removing unwanted objects, correcting imperfections, or, as in this case, turning a striped shirt into a cat wearing a striped T-shirt.

</br>


- **Results with the Inpaint feature applied**

  
   ![Collage-4](https://github.com/chaeyeon2367/genAI-StableDiffusion-VIgeneration/assets/63314860/db0027e1-6036-49ae-a7cb-d0e18559aea7)

</br>


## 4. How to Write Prompts

A prompt is a concise and specific input provided to a generative model to guide its content creation. It typically consists of a brief textual description or set of instructions that influences the output of the model. In the context of Stable Diffusion webui, prompts are used to shape the characteristics, style, or subject matter of the generated images or videos. 

</br>

### (1) Prompt / Negative prompt

Stable Diffusion webui has two input fields for prompts. The first is called the positive prompt, and the second is called the negative prompt. In the positive prompt, you include the content you want reflected in the generated images, while in the negative prompt, you include the content you prefer not to be reflected. However, it's important to note that not everything included in the prompts will be entirely reflected or excluded based on positive or negative prompts.

</br>

### (2) Token

A token, in simple terms, can be thought of as a unit of text, often corresponding to a single character or word. In the context of prompt writing, the number of tokens refers to the count of these text units. The prompt input field in the upper right corner of the Stable Diffusion webui displays the token count. It's recommended to keep prompts within 75 tokens, as the interpretation process divides the text into segments of 75 tokens each. This limitation ensures effective processing and interpretation of the input prompt.

</br>

### (3) Weight

Weight, in simple terms, refers to the influence or impact of a prompt. A prompt without a specified weight is assigned a default weight of 1. You can increase the influence of a prompt by assigning a weight, and there are two ways to do so:

  - **Enclose in Parentheses** : You can enclose the prompt in parentheses to give it additional weight. For example:(best quality)

  - **Colon Notation** : Alternatively, you can use colon notation to explicitly specify the weight. For example: (best quality:1.5)

In this case, a prompt with a weight has a greater impact compared to a prompt without any weight. However, it's essential not to set excessively high weights for a single prompt, as it may negatively affect the generated image. It is generally recommended to set weights within the range of 0.8 to 1.5 to maintain a balance and avoid potential image degradation.

</br>

### (4) Sentence Type / Tag Type Prompts

  - **Sentence Type Prompts** : Sentence type prompts are structured phrases or sentences that provide detailed descriptions in a sentence or clause format. They are ideal for expressing elements like composition, scenario, or actions. Examples include prompts describing appearance, state, background, etc.
    - Example of Sentence Type Prompt: (standing on the table),(looking at window)

  - **Tag Type Prompts** : Tag type prompts consist of single-word prompts that act as concise tags representing specific attributes such as appearance, state, or background. They are more focused and efficient for conveying certain aspects of the desired image.
    - Example of Tag Type Prompt: (black_hair),(white_background)
  
</br>

### (5) Frequently Used Prompts

  - **Prompts**
```
 high quality, 8k, best quality, accurate color reproduction, masterpiece, proportional eyes, same size eyes,
 detailed body,radiosity, realistic, photo-realistic, sharp details, vibrant colors, crystal clear,
 stunning clarity, vivid texture, lifelike rendering, optimal lighting, fine details, rich shadows 
```

  - **Negative Prompts**
```
  7dirtywords, easynegative, worst quality, low quality, extra fingers, fewer fingers,missing fingers,
  extra arms, inaccurate eyes, ugly, deformed, noisy, blurry,low contrast, distorted proportions,
  unrealistic colors, pixelated, dull appearance, unnatural lighting,jagged edges, inconsistent shadows
```

</br>



## 5. ControlNet Variants

  - Different types of ControlNet models and their applications
    
## 6. Dreambooth,LoRA Models Training

  - Dreambooth,LoRA Model Training and results 
  - Differences between Dreambooth and LoRA models



</br>

## 7. Video Generation with Deforum

</br>
Deforum is an open-source animation tool that leverages Stable Diffusion's image-to-image function to produce dynamic video content. The process involves generating a sequence of images and then stitching them together to create a coherent video.

The animation is achieved by applying slight transformations to each frame. The image-to-image function is utilized to generate subsequent frames, ensuring that the transitions between frames are minimal. This approach creates the illusion of smooth continuity, resulting in a visually pleasing and fluid video.

</br>

- **Parameters**

    - **Translation (x, y, z)** : Translation represents the movement of the image in three-dimensional space. Translation x,y,z denotes movement along the x,y,z-axis.

    - **Rotation (3d x, y, z)** : Rotation deals with the orientation or rotation of the image in three-dimensional space. Rotation 3d x,y,z represents rotation around the x,y,z-axis.

    - **Noise Schedule** : The noise schedule refers to a predefined plan or sequence for introducing noise during the generation process. It helps control the randomness or variability in the generated images or video frames. Adjusting the noise schedule can influence the level of detail, texture, or unpredictability in the final output.


  You can find more detailed parameter settings on this website : <https://stable-diffusion-art.com/deforum/>

</br>


- **Superman video generated with Deforum**
    
</br>
 
![superman 2](https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/3f1bf157-bbc6-4dc9-a6d9-e5f93b71c2ae)

</br>


- **Parameters**

  - Translation x: 0: (0), 30:(15), 210:(15), 300:(0)
  - Translation z: 0: (0.2), 60:(10), 300:(15)
  - Rotation 3d x: 0: (0), 60:(0), 90:(0.5), 179:(0), 180:(150), 300:(0.5)
  - Rotation 3d y: 0: (0), 30:(-3.5), 90:(0.5), 180:(-2.8), 300:(-2), 420:(0)
  - Rotation 3d z: 0: (0), 60:(0.2), 90:(0), 180:(-0.5), 300:(0), 420:(0.5), 500:(0.8)
  - Noise schedule: 0: (-0.06*(cos(3.141*t/15)**100)+0.06)

</br>

 - **Prompts**
   
```
  "0": "Superman soaring through the sky, descending to rescue a person, vibrant colors in the background
with clouds and sunlight, Digital illustration, good quality, realistic",
  "60": "Superman descending in an urban environment at night, city lights below creating a dramatic atmosphere,
a mix of tension and relief in the atmosphere, realistic, good quality",
  "120": " Superman descending in a futuristic cityscape, surrounded by holographic displays and advanced technology,
neon lights and advanced architecture, realistic, good quality",
  "180": "Superman descending in a natural setting with a serene landscape, mountains and clear blue sky,
stable diffusion capturing the peaceful yet powerful moment, realistic, good quality, 8k",
  "220":"Superman swooping down towards a person in a chaotic battlefield, smoke and debris in the background,
realistic, good quality, 8k"
```

</br>

    
## 8. Animating Real-human Videos with Move to Move

  - Transforming real human videos into animated sequences

</br>

![1708577651](https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/2a185687-ad20-4d08-98b0-d1ebfae41c53)

</br>


## 9. Video Generation with Animatediff

  - Using Animatediff for generating dynamic and animated videos









