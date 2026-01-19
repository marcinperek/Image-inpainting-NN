import random
from datetime import datetime
from tqdm import trange, tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from ml.UNet import UNet
from ml.Discriminator import Discriminator
from utils.config import load_config
from utils.weights import weights_init_normal
from utils.masking import mask_batch

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

    seed = config["train"]["seed"]
    print("Setup:")
    print("  Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  Using device:", device)

    # Load data
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
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        num_workers=config["train"]["workers"]
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

    criterion = BCELoss()
    real_label = 1.
    fake_label = 0.
    optimizerD = Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerG = Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))

    G_losses = []
    D_losses = []
    iters = 0

    print("\n\nStarting training")

    for epoch in trange(EPOCHS, desc="Epochs"):
        for i, data in enumerate(tqdm(dataloader, desc="Batches", leave=False), 0):
            
            # Update discriminator 

            ## real batch
            netD.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real).view(-1)

            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            ## fake batch
            masked = mask_batch(real)

            fake = netG(masked)

            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)

            lossD_fake = criterion(output, label)
            lossD_fake.backward()

            D_G_z1 = output.mean().item()
            
            lossD = lossD_real + lossD_fake
            optimizerD.step()


            # Update generator
            netG.zero_grad()
            label.fill_(real_label) # for generator loss labels are inverted

            output = netD(fake).view(-1)

            lossG = criterion(output, label)
            lossG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Stats
            if i % 50 == 0:
                tqdm.write('[%d/%d][%d/%d]  Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(dataloader),
                        lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

            
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            iters += 1

        # epoch stats
        tqdm.write(f'Epoch {epoch}/{EPOCHS}: Loss_D: {np.mean(D_losses[:len(dataloader)]):.4f}\tLoss_G: {np.mean(G_losses[:len(dataloader)]):.4f}')

        # save checkpoints
        if epoch % 10 == 0:
            torch.save(netG.state_dict(), f'{MODEL_DIR}/checkpoints/generator_{run}_iter{epoch}.pth')
            torch.save(netD.state_dict(), f'{MODEL_DIR}/checkpoints/discriminator_{run}_iter{epoch}.pth')


    print("\n\nSaving models")
    torch.save(netG.state_dict(), f'{MODEL_DIR}generator_{run}.pth')
    torch.save(netD.state_dict(), f'{MODEL_DIR}discriminator_{run}.pth')

    # save lists G_losses and D_losses
    torch.save(G_losses, f'{MODEL_DIR}G_losses_{run}.pt')
    torch.save(D_losses, f'{MODEL_DIR}D_losses_{run}.pt')