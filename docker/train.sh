#!/bin/sh
mkdir -pv -- "${HOME}/.cache/huggingface/accelerate/"

test -e '/data/input/default_config.yaml' && \
    cp -vf -- '/data/input/default_config.yaml' "${HOME}/default_config.yaml" ;

cp -vf -- "${HOME}/default_config.yaml" \
    "${HOME}/.cache/huggingface/accelerate/default_config.yaml" ;

ln -vfs "${HOME}/GITHUB/aravindhv10/x-flux" "${HOME}/"

cd "${HOME}/x-flux"

if test -e '/data/input/test_lora.yaml'

    then

        test -e '/root/.cache/huggingface/token' || huggingface-cli login
        mkdir -pv -- '/data/output/lora'
        cp -vf -- '/data/input/test_lora.yaml' './train_configs/test_lora.yaml'
        rm -vf -- './images' './lora'
        ln -vfs -- '/data/input' './images'
        ln -vfs -- '/data/output/lora' './'
        accelerate launch \
            './make_latent_2.py' \
        ;

        accelerate launch \
            ./train_flux_lora_controlnet_deepspeed_cached.py \
            --config './train_configs/test_lora.yaml' \
        ;

    else

        echo 'No config file found, copying the sample file and exiting. Edit the example file in the input folder and run again.'

        cp -vf -- './train_configs/test_lora.yaml' '/data/input/test_lora.yaml'

fi
