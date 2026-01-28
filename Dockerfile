FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
python3-dev \
python3-pip \
git \
libglib2.0-0 \
curl \
unzip

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /home/
RUN git clone https://github.com/marcinperek/Image-inpainting-NN.git

WORKDIR /home/Image-inpainting-NN/

RUN /root/.local/bin/uv sync --extra cu126
RUN /root/.local/bin/uv pip install gdown 

RUN ./.venv/bin/gdown https://drive.google.com/uc?id=1gLDZrPMssIJI3kb9jfEL_THbQ9yCKOzD

RUN unzip images.zip
RUN rm images.zip

RUN echo "alias train-unet='uv run --extra cu126 train-unet'" >> ~/.bashrc
RUN echo "alias train-deepfill='uv run --extra cu126 train-deepfill'" >> ~/.bashrc
RUN echo "alias test-unet='uv run --extra cu126 test-unet'" >> ~/.bashrc
RUN echo "alias test-deepfill='uv run --extra cu126 test-deepfill'" >> ~/.bashrc