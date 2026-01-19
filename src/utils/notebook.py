import cv2
import numpy as np
from IPython.display import display
from PIL import Image
from typing import Any
from torch import Tensor
from torchvision.utils import make_grid
from utils.image import image_from_tensor


def imshow(a: np.ndarray[Any, np.dtype[np.uint8]]) -> None:
    """
    Display an image from a numpy array in Jupyter Notebook.

    Image is expected to be either in BGR or BGRA format.

    :param a: Input image array
    :type a: np.ndarray
    """
    a = a.clip(0, 255).astype("uint8")
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA).astype("uint8")
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype("uint8")
    display(Image.fromarray(a))


def display_tensor(img: Tensor) -> None:
    """
    Display an image from a PyTorch tensor in Jupyter Notebook.

    :param img: Input image tensor
    :type img: Tensor
    """
    imshow(image_from_tensor(img))


def display_batch(batch: Tensor | list) -> None:
    """
    Display a batch of images from a PyTorch tensor in Jupyter Notebook.

    Tensor is expected to have shape (B, C, H, W)

    :param batch: Input batch tensor
    :type batch: Tensor
    """
    if type(batch) is list and len(batch) == 2:
        batch = batch[0]

    batch = batch[:25]
    grid = make_grid(batch, padding=4, normalize=True)
    imshow(image_from_tensor(grid))
