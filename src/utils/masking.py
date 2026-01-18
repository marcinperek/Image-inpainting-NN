import numpy as np

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