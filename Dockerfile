FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt install wget git python3 python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 -y && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y
RUN pip3 install --upgrade pip
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
RUN git clone https://github.com/facebookresearch/xformers --recursive && cd xformers && TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.7;8.9" pip install . && cd .. && rm -rf xformers

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install blendmodes accelerate==0.15.0 basicsr fonts font-roboto gfpgan gradio==3.16.2 invisible-watermark numpy omegaconf requests==2.28.2 piexif \
    Pillow pytorch_lightning==1.7.7 realesrgan scikit-image>=0.19 timm==0.6.12 transformers==4.26.0 torch einops==0.6.0 jsonmerge clean-fid resize-right \
    torchdiffeq kornia lark inflection GitPython torchsde safetensors==0.2.6 psutil albumentations==1.3.0 astunparse==1.6.3 beautifulsoup4==4.11.2 importlib-metadata==6.1.0 \
    google-pasta==0.2.0 h5py==3.8.0 bitsandbytes==0.35.0 chardet==4.0.0 clip diffusers[torch]==0.10.2 open-clip-torch lpips==0.1.4 lightning-utilities==0.8.0 ftfy==6.1.1 \
    gast==0.4.0 keras==2.10.0 libclang==15.0.6.1 library opt-einsum==3.3.0 Keras-Preprocessing==1.1.2 joblib==1.2.0 opencv-contrib-python opencv-python==4.7.0.68 gdown==4.6.4 \
    easygui==0.98.3  PySocks==1.7.1   qudida==0.0.4  scikit-learn==1.2.2  sentencepiece==0.1.97  soupsieve==2.4 tensorflow==2.10.1 tensorflow-estimator==2.10.0 \
    tensorflow-io-gcs-filesystem==0.31.0 termcolor==2.2.0 threadpoolctl==3.1.0 toml==0.10.2 voluptuous==0.13.1 wcwidth==0.2.6 wrapt==1.15.0 zipp==3.15.0 \
    tensorboard==2.10.1 altair==4.2.2 fairscale==0.4.13 huggingface-hub==0.12.0 GitPython lion_pytorch

RUN git config --global --add safe.directory /workspace/repositories/stable-diffusion-stability-ai
RUN git config --global --add safe.directory /workspace/repositories/taming-transformers
RUN git config --global --add safe.directory /workspace/repositories/k-diffusion
RUN git config --global --add safe.directory /workspace/repositories/CodeFormer
RUN git config --global --add safe.directory /workspace/repositories/BLIP
RUN git config --global --add safe.directory /workspace/kohya-trainer

EXPOSE 8080
