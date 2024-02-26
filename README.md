

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
6. [Dreambooth LoRA Models Training](#6-Dreambooth-LoRA-Models-Training)
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

</br>

### 📌 What does ControlNet Do?

</br>

<img width="620" alt="Screenshot 2024-02-26 at 17 17 07" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/8867d11b-4180-429f-b61f-6b9cc688d8e3">

</br>
</br>

ControlNet is introduced as a neural network structure designed to augment pretrained large diffusion models, such as Stable Diffusion, by incorporating additional input conditions. The primary purpose of ControlNet is to learn task-specific conditions in an end-to-end manner. Remarkably, the learning process remains robust even when the training dataset is limited, with effectiveness demonstrated even with datasets smaller than 50,000 samples.

ControlNet offers the advantage of efficient training, comparable in speed to fine-tuning a diffusion model. Notably, this training can be performed on personal devices, making it accessible for a broader range of users. Alternatively, if powerful computation clusters are available, ControlNet has the capacity to scale to large datasets, ranging from millions to billions of data points.

The integration of ControlNet with large diffusion models, exemplified by Stable Diffusion, enables the introduction of conditional inputs like edge maps, segmentation maps, keypoints, and more. This capability enriches the methods to control large diffusion models, opening avenues for enhanced control and customization in various applications related to image generation and manipulation.

</br>


### 📌 Different types of ControlNet models

</br>

- **ControlNet Interface**
  
  <img width="512" alt="Screenshot 2024-02-26 at 16 48 53" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/a2d253d9-0b4d-4cff-901c-9c7fcc7c06a2">

</br>

Combining ControlNet models allows for the generation of more customized images based on specific conditions. For instance, using f OpenPose extension in a original image, one can generate a new image with a pose matching that of the person in the original image. This showcases the capability of ControlNet models to leverage different input conditions for creating tailored and desired images.

</br>

- **Released Checkpoints**

  The initial release of ControNet came with the following checkpoints. 

    - [Canny edge](https://huggingface.co/lllyasviel/sd-controlnet-canny) : A monochrome image with white edges on a black background
    - [Depth](https://huggingface.co/lllyasviel/sd-controlnet-depth) : A grayscale image with black representing deep areas and white representing shallow areas
    - [Openpose](https://huggingface.co/lllyasviel/sd-controlnet-openpose) : A OpenPose bone image
    - [Semantic Segmentation Map](https://huggingface.co/lllyasviel/sd-controlnet-seg): An ADE20K's segmentation protocol image
    - [Lineart](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15s2_lineart_anime.pth) : Lineart typically refers to the lines that outline the shapes and forms in an image, often used in illustrations or drawings
    - [Softedge](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_softedge.pth) : Soft edges generally imply smooth transitions between different regions in an image, as opposed to sharp or well-defined edges
 

  Typically, these six ControlNet models are commonly used in practical applications. You can download the checkpoint files for these models from [Hugging Face](https://huggingface.co/lllyasviel/ControlNet-v1-1).

</br>

  - **ReActor**
  
  
    The ReActor Face Swapping Extension in Stable Diffusion is introduced as a robust tool intended to address the absence of Roop. This extension facilitates lifelike face swaps within the Stable Diffusion framework. The comprehensive guide provides instructions on downloading and using the ReActor extension, offering users the capability to achieve realistic face-swapping effects. Additional details and resources can be accessed on the official ReActor [GitHub page](https://github.com/Gourieff/sd-webui-reactor).

      - High-Resolution Face Swaps with Upscaling
      - Efficient CPU Performance
      - Compatibility Across SDXL and 1.5 Models
      - Automatic Gender and Age Detection
      - No NSFW Filter (Uncensored)
      - Continuous Development and Updates


    In summary, the ReActor Extension stands out for high-resolution face swaps with advanced upscaling, optimized for CPU-only setups, offering versatility across SDXL and 1.5 models. It simplifies face-swapping with automatic gender and age detection, supports uncensored swaps, and excels in continuous development for evolving features and advancements in face-swapping technology.

    
</br>

  <img width="720" alt="Screenshot 2024-02-26 at 17 42 19" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/ca4b373e-02b8-45d4-ae60-a797e627b283">

</br>
</br>

  Here is a example of a face swap in Stable Diffusion using Margot Robbie’s face:

</br>


  ![Collage-8](https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/d576069f-5aa8-4b56-a73c-2695e4a7a28e)
  
</br>


🔗 Source : https://ngwaifoong92.medium.com/introduction-to-controlnet-for-stable-diffusion-ea83e77f086e , https://www.nextdiffusion.ai/tutorials/how-to-face-swap-in-stable-diffusion-with-reactor-extension

</br>
</br>

## 6. Dreambooth LoRA Models Training

### 🔎 Dreambooth model 

  - A method of adding new concepts to an already trained model
  - Fine-tunes the weights of the entire model
  - Creates a new checkpoint (weight) as the entire model is modified
  - Occupies a significant amount of disk space, approximately 1-7GB
  - High fidelity to visual features of the subject, preserving existing model knowledge even with fine-tuning using just a few images.
    
</br>
</br>

<img width="860" alt="high_level" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/c1d35334-46e0-452f-b086-d51bd99845a4">


</br>
</br>

The Dreambooth model operates by taking a small set of input images, usually 3-5, depicting a specific subject, along with the corresponding class name (e.g., "dog"). It then produces a fine-tuned or personalized text-to-image model. This model encodes a distinctive identifier specific to the subject. During the inference stage, this unique identifier can be embedded in various sentences to generate synthesized images of the subject in different contexts.


</br>


<img width="860" alt="system" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/4d93ab18-f806-477f-ba64-91d98c6af6e9">


</br>
</br>


The structure involves a two-step fine-tuning process using approximately 3-5 images of a subject:

(a) The initial step fine-tunes a low-resolution text-to-image model using input images paired with a text prompt containing a unique identifier and the class name of the subject (e.g., "A photo of a [T] dog"). Simultaneously, a class-specific prior preservation loss is applied. This loss leverages the model's semantic understanding of the class, encouraging the generation of diverse instances belonging to the subject's class by injecting the class name into the text prompt (e.g., "A photo of a dog").

(b) The subsequent step fine-tunes the super resolution components using pairs of low-resolution and high-resolution images derived from the input image set. This process enables the model to maintain high-fidelity to small details of the subject.

🔗 Source : https://dreambooth.github.io

</br>

### 💡 Dreambooth Model "kami_v02" Training Results

</br>

  ![Collage-7](https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/3639a227-54bf-4b42-aa11-4f6af3079794)

I fine-tuned the Dreambooth model using 20 pictures of our dog, Kami on [Colab](https://colab.research.google.com/drive/1q7UDM2ZM8_RDEH8yfmPeIoXoIkrQjinV). The pre-trained model used for fine-tuning was [realisticVisionV60B1](https://civitai.com/models/4201/realistic-vision-v60-b1). 

 - **Parameters**

    - Crop size : 512
    - Unet_training_steps : 3300
    - Unet_learning_rate : 2e-6
    - Text_encoder_training_steps : 350
    - Text_encoder_learning_rate : 1e-6
    - Resolution : 512

</br>

### 🔎 LoRA model

</br>

  - LoRA introduces subtle changes to the most critical part of the Stable Diffusion model, the cross-attention layer. The cross-attention layer is the point where images and prompts intersect, and even small changes can have significant effects.
  - The modified parts are saved in a separate file and used in conjunction with the ckpt (base model) file.
  - The file size ranges from 2 to 200 MB, relatively smaller compared to the Dreambooth model, and it exhibits decent learning capabilities.
  - The reason behind the smaller file size of the LoRA model, even while storing the same number of weights, lies in its approach of decomposing large matrices into two smaller submatrices with low-rank. In other words, LoRA can store significantly fewer numbers by decomposing matrices into two low-rank matrices.

</br>
    
<img width="673" alt="Screenshot 2024-02-25 at 17 02 01" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/fd999e93-2136-464c-9da2-3fc8c1c58c07">

</br>
</br>

The weights of the cross-attention layer are stored in a matrix. Essentially, a matrix is just an arrangement of numbers organized in rows and columns, similar to an Excel spreadsheet. The LoRA model fine-tunes itself by adding weights to this matrix.

</br>

  - **Weighted Sum**

</br>

<img width="642" alt="Screenshot 2024-02-26 at 15 07 06" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/587069cf-56d7-44a2-bb3a-44983583ad6b">

</br>
</br>

Let's assume a model has a matrix composed of 1000 rows and 2000 columns. In this case, the model file would store 2 million (1000x2000) numbers. LoRA, however, splits this matrix into a 1000x2 matrix and a 2x2000 matrix. This results in only 6000 numbers in total (1000x2 + 2x2000), reducing the size to 1/333 compared to the original matrix. That's why the LoRA file is much smaller.

In this example, the rank of the matrix stored in LoRA is 2, significantly smaller than the original matrix's rank of 2000. This type of reduced-dimension matrix is called a low-rank matrix. However, researchers suggest that reducing the size of the matrix in the cross-attention layer doesn't significantly impact fine-tuning performance. Fortunately, this approach works well.


🔗 Source : [Thesis](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) , https://www.internetmap.kr/entry/How-to-LoRA-Model 

</br>

### 💡 LoRA Model "pkpk" Training Results

</br>

![Collage-9](https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/e165c8ae-84db-4d72-b840-9f38db67ca4d)

</br>

I fine-tuned the LoRA model using 20 pictures of Pikachu on [Colab](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb). The pre-trained model used for fine-tuning was [Chillout-Mix](https://civitai.com/models/6424/chilloutmix).



 - **Parameters**

    - **Data annotation** : BLIP Captioning (batch size 8, max_data_loader_n_workers 2)
    - **Datasets** : resolution = 512, min_bucket_reso = 256, max_bucket_reso = 1024, caption_dropout_rate = 0, caption_tag_dropout_rate = 0, caption_dropout_every_n_epochs = 0, flip_aug = false , color_aug = false
    - **Optimizer_argument** : optimizer_type = "AdamW", learning_rate = 0.0001, max_grad_norm = 1.0, lr_scheduler = "constant", lr_warmup_steps = 0
    - **Training_arguments** : save_precision = "fp16", save_every_n_epochs = 5, train_batch_size = 3, max_token_length = 225, mem_eff_attn = false, xformers = true, max_train_epochs = 25, max_data_loader_n_workers = 8, persistent_data_loader_workers = true, gradient_checkpointing = false, gradient_accumulation_steps = 1, mixed_precision = "fp16", clip_skip = 2

</br>



### 📌 Differences between Dreambooth and LoRA models


  <img width="593" alt="Screenshot 2024-02-25 at 15 50 21" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/8d942bca-c39d-4006-9585-935b76725377">

</br>
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

    
## Animating Real-Person Videos with Move to Move

</br>

Video-to-video tasks are typically labor-intensive and time-consuming, demanding significant manual effort to achieve desired results. The Mov2mov extension, integrated into Stable Diffusion, revolutionizes this workflow by introducing automation to streamline and simplify the entire process. This extension significantly reduces the need for manual intervention, making video-to-video tasks more efficient and accessible to users.

</br>

### ⚙ Enter mov2mov settings

 - Step1 - **Enter a Prompt**

  Next, enter the desired prompt and negative prompt for your video. You can use a detailed description or specific keywords to guide the video generation process.

 - Step2 - **Upload Video**

  Upload the video you wish to work with by dropping it onto the video canvas. Set the Resize mode to: "Crop and resize".

  - Step3 - **Mov2mov Settings**

    <img width="720" alt="Screenshot 2024-02-26 at 18 56 21" src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/0cc1da94-1b33-437c-876b-0c8b8e93ad9b">


    When using the Mov2mov extension, here are some key settings to consider:

      - **Sampling method** : Keep in mind that deterministic samplers like Euler, LMS, and DPM++2M Karras might not work well with this extension, as they may not effectively reduce flickering.
     - **Noise Multiplier** : Utilize the slider to adjust the noise multiplier. For smoother results and reduced flickering, keep it at 0.
     - **CFG Scale** : Control the extent to which the prompt is followed by adjusting the CFG scale. In the provided video, a scale of 7 was used.
     - **Denoising Strength** : Fine-tune the amount of change applied to the video by adjusting the denoising strength. A value of 0.6 was used in the example video.
     - **Movie Frames** : The frames per second of your output. The higher this value, the smoother your video, but this will have to render more images.
     - **Max Frame** : Determine the total number of frames to be generated. For initial testing, set it to a low number such as 10. To generate a full-length video, set it to -1.
    - **Seed** : The seed determines the value for the first frame. All frames will use the same seed value, even if you set it to -1 for a random seed.

</br>


### 📍Generate videos with mov2mov extension

With all the settings in place, it's time to generate the video. Click the "Generate" button to start the process. Be patient as it may take some time. Once the generation is complete, your new video will appear on the right side of the page.

Click "Save" to download and save the video to your device. If you can't locate the video, check the output/mov2mov-videos folder.

</br>

  <p float="left">
    <img src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/2a185687-ad20-4d08-98b0-d1ebfae41c53" height="620"/>
    <img src="https://github.com/chaeyeon2367/genAI-StableDiffusion-IVgeneration/assets/63314860/e6a0b65e-5092-4cc1-af92-c7bcd6025217" height="620"/> 
  </p>

</br>

🔗 Source : https://www.nextdiffusion.ai/tutorials/transforming-videos-into-stunning-ai-animations-with-stable-diffusion-mov2mov
🔗 Original Videos : https://youtube.com/shorts/4cT2swoyNAY?si=B_OwxpMP-D2msK7I , https://youtube.com/shorts/2yZRp7wcqKk?si=FWDIucIFD7xG5cRG


## 9. Video Generation with Animatediff

  - Using Animatediff for generating dynamic and animated videos









