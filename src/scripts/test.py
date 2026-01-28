import random
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from ml.UNet import UNet
from ml.DeepFill import DeepFill
from utils.config import load_config
from utils.masking import mask_batch
from utils.losses import VGGLoss


def test_deepfill():
    config = load_config("config_test.toml")
    IMG_SIZE = config["files"]["image_size"]

    random.seed(config["test"]["seed"])
    torch.manual_seed(config["test"]["seed"])
    np.random.seed(config["test"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(
        root=config["files"]["data_root"],
        transform=transforms
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=config["test"]["batch_size"], 
        shuffle=True, 
        num_workers=config["test"]["workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )

    deepfill = DeepFill(device, cnum=config["model"]["cnum"]).to(device)
    deepfill.load_state_dict(torch.load(config["model"]["weights_path"], map_location=device))
    deepfill.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    vgg = VGGLoss(device)

    avg_psnr_coarse = 0.0
    avg_ssim_coarse = 0.0
    avg_L1_coarse = 0.0
    avg_vgg_coarse = 0.0

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_L1 = 0.0
    avg_vgg = 0.0

    for i, data in enumerate(tqdm(dataloader, desc="Testing DeepFill")):
        batch = data[0].to(device)
        masked_batch = mask_batch(batch, device)
        with torch.no_grad():
            coarse, output = deepfill(masked_batch)
        
        avg_L1_coarse += L1Loss()(coarse, batch).item()
        avg_vgg_coarse += vgg(coarse, batch).item()
        avg_psnr_coarse += psnr(coarse, batch).item()
        avg_ssim_coarse += ssim(coarse, batch).item()

        avg_L1 += L1Loss()(output, batch).item()
        avg_vgg += vgg(output, batch).item()
        avg_psnr += psnr(output, batch).item()
        avg_ssim += ssim(output, batch).item()

    avg_L1_coarse /= len(dataloader)
    avg_vgg_coarse /= len(dataloader)
    avg_psnr_coarse /= len(dataloader)
    avg_ssim_coarse /= len(dataloader)

    avg_L1 /= len(dataloader)
    avg_vgg /= len(dataloader)
    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)

    print(f"Average L1 Loss: {avg_L1:.4f}")
    print(f"Average VGG Loss: {avg_vgg:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    print(f"Average L1 Loss (coarse): {avg_L1_coarse:.4f}")
    print(f"Average VGG Loss (coarse): {avg_vgg_coarse:.4f}")
    print(f"Average PSNR (coarse): {avg_psnr_coarse:.4f} dB")
    print(f"Average SSIM (coarse): {avg_ssim_coarse:.4f}")


def test_unet():
    config = load_config("config_test.toml")
    IMG_SIZE = config["files"]["image_size"]

    random.seed(config["test"]["seed"])
    torch.manual_seed(config["test"]["seed"])
    np.random.seed(config["test"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(
        root=config["files"]["data_root"],
        transform=transforms
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=config["test"]["batch_size"], 
        shuffle=True, 
        num_workers=config["test"]["workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )

    unet = UNet(device).to(device)
    unet.load_state_dict(torch.load(config["model"]["weights_path"], map_location=device))
    unet.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    vgg = VGGLoss(device)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_L1 = 0.0
    avg_vgg = 0.0

    for i, data in enumerate(tqdm(dataloader, desc="Testing UNet")):
        batch = data[0].to(device)
        masked_batch = mask_batch(batch, device)
        with torch.no_grad():
            output = unet(masked_batch)
        avg_psnr += psnr(output, batch).item()
        avg_ssim += ssim(output, batch).item()
        avg_L1 += L1Loss()(output, batch).item()
        avg_vgg += vgg(output, batch).item()

    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)
    avg_L1 /= len(dataloader)
    avg_vgg /= len(dataloader)

    print(f"Average L1 Loss: {avg_L1:.4f}")
    print(f"Average VGG Loss: {avg_vgg:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
