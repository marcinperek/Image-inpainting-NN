# Image inpainting project
This repository is a part of a project for Computer Vision course at Politechnika Poznańska.

The goal of this project is to train and evaluate an image inpainting model.


## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/marcinperek/Image-inpainting-NN.git
cd Image-inpainting-NN
uv sync --extra [torch-version]
```
where `[torch-version]` can be set to:
- `cpu` for CPU-only version
- `cu126` for CUDA 12.6 support


## Dataset
Download the Places365-Standard dataset from: http://places2.csail.mit.edu/download.html.<br>
Modify the `BASE_DIR` variable in `make_dataset.py` to point to the downloaded dataset path and run the script to create training and testing datasets.

## Train
Train UNet with discriminator loss:

```bash
uv run [--extra cu126] train-unet
```

Or train DeepFill with PatchGAN discriminator loss:

```bash
uv run [--extra cu126] train-deepfill
```

If using CUDA pass `--extra cu126` to the run command.

Training configuration can be changed in `config.toml` for UNet or `config_deepfill.toml` for DeepFill.

## Test
To test model performance run:

```bash
uv run [--extra cu126] test-[unet|deepfill]
```

Test configuration can be changed in `test_config.toml`.

## Weights
Weights can be downloaded here: https://drive.google.com/drive/folders/1HfID7NYkO2YLtg5oFcNX8ENMkjZmfXQG?usp=sharing


## Alternative installation - Docker
Alternatively, you can use Docker to set up the environment. Build and run the Docker image with:

```bash
docker build -t image-inpainting-nn .
docker run -it --gpus all image-inpainting-nn bash
```

To run the training or testing scripts inside the Docker container, simply use one of the following commands:

```bash
train-unet
train-deepfill
test-unet
test-deepfill
```

All other instructions remain the same as for normal installation.