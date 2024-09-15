#!/bin/sh
mkdir -pv -- "${2}"

IMAGE_NAME='flux_train_xflux'
CONTAINER_NAME="${IMAGE_NAME}_1"

sudo docker run \
    --tty \
    --interactive \
    --rm \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(realpath "${1}"):/data/input" \
    -v "$(realpath "${2}"):/data/output" \
    -v "CACHE:/root/.cache" \
    "${IMAGE_NAME}" \
    /bin/bash \
;
