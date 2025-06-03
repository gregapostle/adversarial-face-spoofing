import numpy as np
from PIL import Image
import cv2
import torch

def apply_jpeg_compression(img_tensor, quality=75):
    """
    Simulate saving and reloading image using JPEG compression.
    """
    img = tensor_to_pil(img_tensor)
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    return pil_to_tensor(compressed)

def apply_gaussian_blur(img_tensor, ksize=3):
    """
    Apply Gaussian blur to smooth out small perturbations.
    """
    img = tensor_to_numpy(img_tensor)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return numpy_to_tensor(blurred)

# === Tensor <--> Image Conversion Helpers ===

def tensor_to_pil(tensor):
    array = (tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)

def pil_to_tensor(img):
    img = img.convert('RGB')
    array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor

def tensor_to_numpy(tensor):
    return tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

def numpy_to_tensor(array):
    array = np.clip(array, 0, 1).astype(np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor
