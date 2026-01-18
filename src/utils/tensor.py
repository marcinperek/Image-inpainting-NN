import numpy as np
from torch import Tensor
from typing import Any


def numpy_to_tensor(img: np.ndarray[Any, np.dtype[np.uint8]]) -> Tensor:
    """
    Convert image in NumPy array format to PyTorch tensor.

    NumPy array is expected to have shape (H, W, C) and dtype uint8.
    Tensor will have shape (C, H, W) and dtype float32.

    :param img: Input NumPy array
    :type img: np.ndarray
    :return tensor: Converted tensor
    """
    return Tensor(img.astype(np.float32).transpose(2, 0, 1))


def tensor_to_numpy(tensor: Tensor) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """
    Convert PyTorch tensor after sigmoid back to image in NumPy array format.

    Tensor is expected to have shape (C, H, W) and be on GPU device.
    NumPy array will have shape (H, W, C) and dtype uint8.

    :param tensor: Input tensor
    :type tensor: Tensor
    :return img: Converted NumPy array
    """
    res = tensor.permute(1, 2, 0).cpu().detach().numpy()
    return (res).astype(np.uint8)
