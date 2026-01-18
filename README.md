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