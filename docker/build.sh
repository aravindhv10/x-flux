#!/bin/sh
cd "$('dirname' '--' "${0}")"

IMAGE_NAME='flux_train_xflux'
CONTAINER_NAME="${IMAGE_NAME}_1"

sudo docker image build \
    -t "${IMAGE_NAME}" \
    . \
;
