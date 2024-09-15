FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV HOME='/root'
ENV DEBIAN_FRONTEND='noninteractive'

RUN apt-get update && apt-get install -y \
    git \
    aria2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
;

RUN conda install --yes \
    conda-forge::timm \
    pytorch::torchvision \
    conda-forge::onnxruntime \
    conda-forge::matplotlib \
    conda-forge::opencv \
    conda-forge::sentencepiece \
    conda-forge::omegaconf \
;

RUN --mount=type=cache,target=/root/.cache pip install \
    accelerate==0.30.1 \
    deepspeed==0.14.4 \
    einops==0.8.0 \
    transformers==4.43.3 \
    huggingface-hub==0.24.5 \
    optimum-quanto \
    datasets \
    diffusers \
    prodigyopt \
;

COPY ./important_functions.sh /root/important_functions.sh

# RUN . "${HOME}/important_functions.sh" ; \
#     get_repo 'https://github.com/XLabs-AI/x-flux.git' main ;

RUN . "${HOME}/important_functions.sh" ; \
    get_repo 'https://github.com/aravindhv10/x-flux/' 'aravind_prodigy_dataset' '54ed13998360f2ef4ae53bda044140b4f434c84a' ; 


RUN mkdir -pv -- /data/input /data/output

COPY './default_config.yaml' '/root/default_config.yaml'
COPY './train.sh' '/root/train.sh'
