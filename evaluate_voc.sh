#!/usr/bin/bash

python evaluate.py \
    --network mobilenet_yolo \
    --rec-path ./data/VOCdevkit/val.rec \
    --num-class 20 \
    --class-names ./dataset/names/pascal_voc.names \
    --prefix ./model/yolo2_symbol_mobilenet_416 \
    --epoch 151 \
    --data-shape 416 \
    --nms 0.35 \
    --gpus 0,1
