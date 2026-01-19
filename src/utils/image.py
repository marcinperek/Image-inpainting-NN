from torch import Tensor
from torch import clip
import numpy as np
from utils.tensor import tensor_to_numpy

def rescale_tensor(img:Tensor) -> Tensor:
    """
    Rescale normalized image from either [-1, 1] or [0, 1] to [0, 255] uint8 range
    """
    if img.min() < 0:
        img = (img + 1) / 2
    img = clip(img, 0, 1)
    return img * 255

def image_from_tensor(img: Tensor) -> np.ndarray:
    img = rescale_tensor(img)
    return tensor_to_numpy(img, bgr=True)