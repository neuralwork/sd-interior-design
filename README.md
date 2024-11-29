## RealisticVision for Interior Design
A custom interior design pipeline API that combines [Realistic Vision V3.0](https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE) inpainting pipeline with [segmentation](https://huggingface.co/BertChristiaens/controlnet-seg-room) and [MLSD](https://huggingface.co/lllyasviel/sd-controlnet-mlsd) ControlNets. This repo uses [Cog](https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md) to create a dockerized API. See the Replicate [demo](https://replicate.com/adirik/interior-design) to test the running API.


## Basic Usage
You will need to have [Cog](https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md) and Docker installed to serve your model as an API. To run a prediction:

```bash
cog predict -i image=@test_images/bedroom_3.jpg prompt="A bedroom with a bohemian spirit centered around a relaxed canopy bed complemented by a large macrame wall hanging. An eclectic dresser serves as a unique storage solution while an array of potted plants brings life and color to the room"
```

To start your server and serve the model as an API:
```bash
cog run -p 5000 python -m cog.server.http
```

The API input arguments are as follows:

- **image:** The provided image serves as a base or reference for the generation process.  
- **prompt:** The input prompt is a text description that guides the image generation process. It should be a detailed and specific description of the desired output image.  
- **negative_prompt:** This parameter allows specifying negative prompts. Negative prompts are terms or descriptions that should be avoided in the generated image, helping to steer the output away from unwanted elements.  
- **num_inference_steps:** This parameter defines the number of denoising steps in the image generation process.  
- **guidance_scale:** The guidance scale parameter adjusts the influence of the classifier-free guidance in the generation process. Higher values will make the model focus more on the prompt.  
- **prompt_strength:** In inpainting mode, this parameter controls the influence of the input prompt on the final image. A value of 1.0 indicates complete transformation according to the prompt.  
- **seed:** The seed parameter sets a random seed for image generation. A specific seed can be used to reproduce results, or left blank for random generation.  

## Model Details

This is a custom pipeline inspired by AICrowd's Generative Interior Design [hackathon](https://www.aicrowd.com/challenges/generative-interior-design-challenge-2024) that uses [Realistic Vision V3.0](https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE) as the base model. See the base and ControlNet model pages for their respective licenses. This code base is licensed under the [MIT license](https://github.com/neuralwork/sd-interior-design/blob/main/LICENSE).

<a href='https://ko-fi.com/Z8Z616R4PF' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

From [neuralwork](https://neuralwork.ai/) with :heart:
