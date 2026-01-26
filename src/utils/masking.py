import numpy as np
from torch import Tensor
import torch

def mask_image(img: np.ndarray) -> np.ndarray:
    width, height = img.shape[1], img.shape[0]
    padding = 30
    min_size = 30
    max_size = 80

    box_width = np.random.randint(min_size, max_size)
    box_height = np.random.randint(min_size, max_size)

    anchor = np.random.randint(0+padding, width - padding - box_width), np.random.randint(0+padding, height - padding - box_height)


    img[anchor[1]:anchor[1]+box_height, anchor[0]:anchor[0]+box_width, :] = 0

    return img

def mask_batch(batch: Tensor, value=1) -> Tensor:
    masked_batch = batch.clone()
    batch_size, channels, height, width = batch.shape
    padding = 30
    min_size = 30
    max_size = 80

    for i in range(batch_size):
        box_width = np.random.randint(min_size, max_size)
        box_height = np.random.randint(min_size, max_size)

        anchor = np.random.randint(0+padding, width - padding - box_width), np.random.randint(0+padding, height - padding - box_height)

        masked_batch[i, :, anchor[1]:anchor[1]+box_height, anchor[0]:anchor[0]+box_width] = value

    return masked_batch

def generate_batch_masks(batch: Tensor ,device, seed: int | None = None) -> Tensor:
    """
    Generates random rectangular masks for each image in the batch.
    """
    batch_size, channels, height, width = batch.shape
    padding = 30
    min_size = 30
    max_size = 80

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    box_widths = torch.randint(min_size, max_size, (batch_size,), device=device, generator=gen)
    box_heights = torch.randint(min_size, max_size, (batch_size,), device=device, generator=gen)

    range_x = width - padding - box_widths - padding
    range_y = height - padding - box_heights - padding

    anchor_x = (torch.rand(batch_size, device=device, generator=gen) * range_x).long() + padding
    anchor_y = (torch.rand(batch_size, device=device, generator=gen) * range_y).long() + padding

    return torch.stack([anchor_x, anchor_y, box_widths, box_heights], dim=1)

def get_mask_channel(batch: Tensor, masks: Tensor, device):
    """
    Generates a binary mask channel for the batch based on the provided masks.
    """
    mask_batch = torch.zeros_like(batch).to(device)
    batch_size, channels, height, width = batch.shape
    anchor_x = masks[:, 0]
    anchor_y = masks[:, 1]
    box_widths = masks[:, 2]
    box_heights = masks[:, 3]


    for i in range(batch_size):
        y1, x1 = anchor_y[i].item(), anchor_x[i].item()
        h, w = box_heights[i].item(), box_widths[i].item()
        
        mask_batch[i, :, y1:y1+h, x1:x1+w] = 1.0

    return mask_batch

def overlay_masks(batch: Tensor, masks: Tensor, device) -> Tensor:
    """
    Overlays masks on the batch by setting the masked regions to -1.0.
    """
    masked_batch = batch.clone()
    batch_size, channels, height, width = batch.shape
    anchor_x = masks[:, 0]
    anchor_y = masks[:, 1]
    box_widths = masks[:, 2]
    box_heights = masks[:, 3]


    for i in range(batch_size):
        y1, x1 = anchor_y[i].item(), anchor_x[i].item()
        h, w = box_heights[i].item(), box_widths[i].item()
        
        masked_batch[i, :, y1:y1+h, x1:x1+w] = -1.0

    return masked_batch