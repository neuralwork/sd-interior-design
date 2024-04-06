# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
from typing import Tuple, Union, List

import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet import StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from colors import ade_palette
from utils import map_colors_rgb

from cog import BasePredictor, Input, Path


def filter_items(
    colors_list: Union[List, np.ndarray],
    items_list: Union[List, np.ndarray],
    items_to_remove: Union[List, np.ndarray],
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
            
    return filtered_colors, filtered_items


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        controlnet = [
            ControlNetModel.from_pretrained(
                "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
            ),
        ]

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V3.0_VAE",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe = self.pipe.to("cuda")

        self.control_items = [
            "windowpane;window",
            "column;pillar",
            "door;double;door",
        ]
        self.additional_quality_suffix = "interior design, 4K, high resolution, elegant, tastefully decorated, functional"
        self.seg_image_processor = image_processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        self.image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        self.mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        

    @torch.inference_mode()
    @torch.autocast("cuda")
    def segment_image(self, image):
        """
        Segments an image using a semantic segmentation model.

        Args:
            image (PIL.Image): The input image to be segmented.
            image_processor (AutoImageProcessor): The processor to prepare the
                image for segmentation.
            image_segmentor (SegformerForSemanticSegmentation): The semantic
                segmentation model used to identify different segments in the image.

        Returns:
            Image: The segmented image with each segment colored differently based
                on its identified class.
        """
        pixel_values = self.seg_image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)

        seg = self.seg_image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
            
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert("RGB")
        
        return seg_image

    def resize_dimensions(self, dimensions, target_size):
        """
        Resize PIL to target size while maintaining aspect ratio
        If smaller than target size leave it as is
        """
        width, height = dimensions

        # Check if both dimensions are smaller than the target size
        if width < target_size and height < target_size:
            return dimensions

        # Determine the larger side
        if width > height:
            # Calculate the aspect ratio
            aspect_ratio = height / width
            # Resize dimensions
            return (target_size, int(target_size * aspect_ratio))
        else:
            # Calculate the aspect ratio
            aspect_ratio = width / height
            # Resize dimensions
            return (int(target_size * aspect_ratio), target_size)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Text prompt for design"),
        negative_prompt: str = Input(
            description="Negative text prompt to guide the design",
            default="lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, extra, ugly, upholstered walls, fabric walls, plush walls, mirror, mirrored, functional, realistic",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=15
        ),
        prompt_strength: float = Input(
            description="Prompt strength for inpainting. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
            
        img = Image.open(str(image))
        
        if "bedroom" in prompt and "bed " not in prompt:
            prompt += ", with a queen size bed against the wall"
        elif "children room" in prompt or "children's room" in prompt:
            if "bed " not in prompt:
                prompt += ", with a twin bed against the wall"

        pos_prompt = prompt + f", {self.additional_quality_suffix}"

        orig_w, orig_h = img.size
        new_width, new_height = self.resize_dimensions(img.size, 768)
        input_image = img.resize((new_width, new_height))

        # preprocess for segmentation controlnet
        real_seg = np.array(
            self.segment_image(input_image)
        )
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=self.control_items,
        )
        mask = np.zeros_like(real_seg)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1

        image_np = np.array(input_image)
        image = Image.fromarray(image_np).convert("RGB")
        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

        # preprocess for mlsd controlnet
        mlsd_img = self.mlsd_processor(input_image)
        mlsd_img = mlsd_img.resize(image.size)

        generated_image = self.pipe(
            prompt=pos_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=[torch.Generator(device="cuda").manual_seed(seed)],
            image=image,
            mask_image=mask_image,
            control_image=[segmentation_cond_image, mlsd_img],
            controlnet_conditioning_scale=[0.4, 0.2],
            control_guidance_start=[0, 0.1],
            control_guidance_end=[0.5, 0.25],
        ).images[0]

        out_img = generated_image.resize(
            (orig_w, orig_h), Image.Resampling.LANCZOS
        )
        
        out_img = out_img.convert("RGB")
        out_path = "out.png"
        out_img.save(out_path)
        
        return Path(out_path)
