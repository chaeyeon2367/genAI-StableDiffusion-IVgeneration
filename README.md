
</br>

# 🌈AI Image and Video Generation Project with Stable Diffusion

</br> 

## 💡 Project Overview
This repository showcases the results of an AI image and video generation project using Stable Diffusion. The project involves utilizing the Stable Diffusion WebUI for Prompt Engineering, ControlNet, Dreambooth, and LoRA Generative AI models. Dreambooth and LoRA models have been trained on a custom dataset,and the generated content is included in this repository.

</br> 

## 🏷 Table of Contents

1. [Stable Diffusion Introduction and WebUI Installation](#1-stable-diffusion-introduction-and-webui-installation)
2. [Text to Image (t2i)](#2-text-to-image)
3. [Image to Image (i2i)](#3-image-to-image)
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

### 📌 Install Stable Diffusion Webui on Colab and Locally

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

       Open a web browser and click the following URL to start Stable Diffusion.

        `http://127.0.0.1:7860/`

#### (2) Install stable-diffusion-webui on Colab

  - Version 1. [AUTOMATIC1111](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)
  - Version 2. [digiclau](https://colab.research.google.com/github/DigiClau/stablediffusion_webui/blob/main/StableDiffusionWebUI_digiclau.ipynb) korean ver.

</br> 

### 📌 Stable Diffusion models(checkpoint)

The Stable Diffusion model, akin to the artist who draws in the space provided by the Stable Diffusion webui, is the entity responsible for creating images. In other words, choosing a model is comparable to selecting the artist who will be drawing. While techniques like LoRA, embedding, hypernetwork, and others are capable of generating images, the model or checkpoint serves as the artist; without it, there is no one to create the artwork. Therefore, having the right model (checkpoint) is essential for the generation of images in the Stable Diffusion framework.

In the same way, choosing a model is akin to selecting the artist, and just as the style of an artwork varies depending on who is drawing, the images generated differ significantly based on the choice of the model. There are two main websites where you can download these models.

  - Civitai : <https://civitai.com>

  - Hugging face : <https://huggingface.co>

</br> 

#### Model files (.safetensors , .ckpt)

The model files with the extensions .safetensors and .ckpt are related to the Stable Diffusion webui and represent different aspects of the model:

   - safetensors : This file contains tensors (data structures representing multi-dimensional arrays) related to the model.
It may include information about the model's architecture, parameters, and other essential components.
The ".safetensors" extension suggests that the data stored in this file is considered safe or stable for the model's functioning.

   - 'ckpt' : This file typically represents a checkpoint file and contains the saved weights and biases of the model.
It allows the model to be saved and restored at a later time, enabling users to continue training or deploy the model without starting from scratch.
The ".ckpt" extension is a common convention in machine learning to denote checkpoint files.


## 2. Text to Image (t2i)
  - Results of Text to Image generation

    
## 3. Image to Image (i2i)
  - Results of Image to Image generation

## 4. How to Write Prompts
    
## 5. ControlNet Variants

  - Different types of ControlNet models and their applications
    
## 6. Dreambooth,LoRA Models Training

  - Dreambooth,LoRA Model Training and results 
  - Differences between Dreambooth and LoRA models

## 7. Video Generation with Deforum

  - Using Deforum for generating AI videos
    
## 8. Animating Real-human Videos with Move to Move

  - Transforming real human videos into animated sequences

## 9. Video Generation with Animatediff

  - Using Animatediff for generating dynamic and animated videos












Follow the steps below to set up the project locally:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
# Add instructions for any specific setup steps if necessary

