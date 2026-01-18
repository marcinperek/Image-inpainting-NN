import cv2
import numpy as np
from IPython.display import display
from PIL import Image
from typing import Any


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