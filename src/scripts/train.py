import random
from datetime import datetime
from tqdm import trange, tqdm
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch.nn import BCELoss, L1Loss
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
from torchvision.utils import save_image, make_grid

from ml.UNet import UNet
from ml.Discriminator import Discriminator
from utils import config
from utils.config import load_config
from utils.weights import weights_init_normal
from utils.masking import mask_batch, mask_batch_torch

def train():
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Initial setup
    config = load_config("config.toml")
    IMG_SIZE = config["files"]["image_size"]
    LR = config["train"]["learning_rate"]
    BETA1 = config["train"]["beta1"]
    BETA2 = config["train"]["beta2"]
    EPOCHS = config["train"]["epochs"]
    MODEL_DIR = config["files"]["output_dir"]

    wandb.init(
        entity="marcin-and-adam",
        project="image-inpainting",
        name=f"run_{run}",
        config=config
    )

    seed = config["train"]["seed"]
    print("Setup:")
    print("  Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  Using device:", device)

    # data augmentation
    train_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(10),
        RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # loading datasets sepately to apply different transforms
    train_dataset_full = ImageFolder(
        root=config["files"]["data_root"],
        transform=train_transforms
    )
    
    val_dataset_full = ImageFolder(
        root=config["files"]["data_root"],
        transform=val_transforms
    )

    num_train = int(len(train_dataset_full) * 0.9)
    indices = list(range(len(train_dataset_full)))

    np.random.shuffle(indices) 
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        num_workers=config["train"]["workers"],
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize models
    netG = UNet(device).to(device)
    netG.apply(weights_init_normal)
    print("\n\nGenerator architecture:")
    netG.summary()

    netD = Discriminator(device, hidden_dim=config["model"]["hidden_channels"]).to(device)
    netD.apply(weights_init_normal)
    print("\n\nDiscriminator architecture:")
    netD.summary()

    # criterion = BCELoss()
    # real_label = 1.
    # fake_label = 0.
    criterion_GAN = BCELoss()
    criterion_pixel = L1Loss()
    
    real_label = 0.9 
    fake_label = 0.0
    
    lambda_pixel = 100.0

    optimizerD = Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerG = Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))

    G_losses = []
    D_losses = []
    iters = 0

    print("\n\nStarting training")

    for epoch in trange(EPOCHS, desc="Epochs"):
        netG.train()
        netD.train()

        for i, data in enumerate(tqdm(train_loader, desc="Batches", leave=False), 0):
            real = data[0].to(device)
            b_size = real.size(0)

            # Update discriminator 

            netD.zero_grad()

            ## real batch
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(real).view(-1)
            lossD_real = criterion_GAN(output_real, label)
            lossD_real.backward()
            D_x = output_real.mean().item()

            ## fake batch
            masked = mask_batch_torch(real, device)

            fake = netG(masked)
            label.fill_(fake_label)
            output_fake = netD(fake.detach()).view(-1)

            lossD_fake = criterion_GAN(output_fake, label)
            lossD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            
            lossD = lossD_real + lossD_fake
            optimizerD.step()


            # Update generator
            netG.zero_grad()
            label.fill_(real_label) # for generator loss labels are inverted

            output = netD(fake).view(-1)

            loss_adv = criterion_GAN(output, label)
            loss_pixel = criterion_pixel(fake, real)
            # lossG = criterion(output, label)
            lossG = loss_adv + lambda_pixel * loss_pixel
            lossG.backward()
            optimizerG.step()

            D_G_z2 = output.mean().item()

            # Stats
            if i % 50 == 0:
                tqdm.write('[%d/%d][%d/%d]  Loss_D: %.4f\tLoss_G: %.4f\tLoss_adv: %.4f\tLoss_pixel: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(train_loader),
                        lossD.item(), lossG.item(), loss_adv.item(), loss_pixel.item(), D_x, D_G_z1, D_G_z2))
                
                # log to wandb
                wandb.log({
                    "train/loss_D": lossD.item(),
                    "train/loss_G": lossG.item(),
                    "train/loss_adv": loss_adv.item(),
                    "train/loss_pixel": loss_pixel.item(),
                    "train/D_x": D_x,
                    "train/D_G_z": D_G_z1
                })

            
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            iters += 1

        # validation
        netG.eval() 
        val_loss = 0.0
        VAL_SEED = 555 
        
        with torch.no_grad():
            for batch_idx, (real_val, _) in enumerate(val_loader):
                current_seed = VAL_SEED + batch_idx
                real_val = real_val.to(device)
                masked_val = mask_batch_torch(real_val, device, seed=current_seed)
                fake_val = netG(masked_val)

                val_loss += criterion_pixel(fake_val, real_val).item()
                
                if batch_idx == 0:
                    logging_real = real_val[:4]
                    logging_masked = masked_val[:4]
                    logging_fake = fake_val[:4]

        avg_val_loss = val_loss / len(val_loader)

        # log to wandb
        log_dict = {"val/l1_loss": avg_val_loss, "epoch": epoch}
        # epoch stats
        tqdm.write(f'Epoch {epoch}/{EPOCHS}: Loss_D: {np.mean(D_losses[-len(train_loader):]):.4f}\tLoss_G: {np.mean(G_losses[-len(train_loader):]):.4f}| Val L1 Loss: {avg_val_loss:.4f}')

        # save checkpoints and logging images every 5 epochs
        if epoch % 5 == 0:
            logging_images = torch.cat((logging_real, logging_masked, logging_fake), dim=2)
            logging_images = (logging_images + 1) / 2.0  # denormalization
            grid = make_grid(logging_images, nrow=1)

            log_dict["val/log_images"] = wandb.Image(grid, caption=f"Epoch {epoch} Results")

            torch.save(netG.state_dict(), f'{MODEL_DIR}/checkpoints/generator_{run}_iter{epoch}.pth')
            torch.save(netD.state_dict(), f'{MODEL_DIR}/checkpoints/discriminator_{run}_iter{epoch}.pth')

        wandb.log(log_dict)

    print("\n\nSaving models")
    torch.save(netG.state_dict(), f'{MODEL_DIR}generator_{run}.pth')
    torch.save(netD.state_dict(), f'{MODEL_DIR}discriminator_{run}.pth')

    # save lists G_losses and D_losses
    torch.save(G_losses, f'{MODEL_DIR}G_losses_{run}.pt')
    torch.save(D_losses, f'{MODEL_DIR}D_losses_{run}.pt')
    
    wandb.finish()