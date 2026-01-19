import os

from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.v2 import RGB

from utils.notebook import imshow
from utils.tensor import tensor_to_numpy


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = decode_image(img_path)
        img = RGB()(img)  # some images were in grayscale (417, 738, ...)
        if self.transform:
            img = self.transform(img)
        # img = img.float() / 255.0 
        return img

    def get_filename(self, idx):
        return self.img_names[idx]

    def show_image(self, idx):
        img = self.__getitem__(idx)
        img_np = tensor_to_numpy(img)
        img_np[:, :, [0, 1, 2]] = img_np[:, :, [2, 1, 0]]  # convert RGB to BGR
        imshow(img_np)
