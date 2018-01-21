#!/usr/bin/bash

python evaluate_wider.py \
    --network mobilenet_yolo \
    --dataset pascal_voc \
    --image-set test \
    --year 2007 \
    --devkit-path ./data/VOCdevkit \
    --data-shape 576 \
    --prefix ./model/yolo2_mobilenet \
    --epoch 118 \
    --gpus 1
