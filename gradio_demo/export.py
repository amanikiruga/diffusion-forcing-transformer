from typing import List
import tempfile
import numpy as np
import torch
from torch import Tensor
from torchvision.io import write_video
from PIL import Image
import imageio.v3 as iio

# def export_to_video(tensor: Tensor, fps: int = 10) -> str:
#     path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
#     video = (
#         (tensor.permute(0, 2, 3, 1) * 255)
#         .clamp(0, 255)
#         .to(torch.uint8)
#         .contiguous()  # ✅ this line is key
#     )
#     write_video(path, video, fps=fps)
#     return path



def export_to_video(tensor: Tensor, fps: int = 10) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    video = (
        tensor.permute(0, 2, 3, 1)
        .mul(255)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    iio.imwrite(
        path,
        video,
        fps=fps,
        plugin="pyav",        # Or "ffmpeg" if you have it
        codec="libx264",      # ✅ Explicitly set codec
    )
    return path
def export_to_gif(tensor: Tensor, fps: int = 4) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".gif").name
    images = (tensor.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    images = [Image.fromarray(image.numpy()) for image in images]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )
    return path

def export_images_to_gif(images: List[np.ndarray], fps: int = 4) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".gif").name
    images = [Image.fromarray(image) for image in images]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )
    return path