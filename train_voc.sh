python train.py \
    --train-path ./data/VOCdevkit/train.rec \
    --val-path ./data/VOCdevkit/val.rec \
    --num-class 20 \
    --class-names ./dataset/names/pascal_voc.names \
    --network symbol_mobilenet_yolo \
    --label-width 350 \
    --batch-size 32 \
    --data-shape 416 \
    --min-random-shape 320 \
    --max-random-shape 512 \
    --random-shape-step 32 \
    --freeze '^(conv1|conv2).*' \
    --optimizer-name gnadam \
    --lr 5e-03 \
    --wd 5e-04 \
    --lr-steps 80,110 \
    --lr-factor 0.1 \
    --pretrained /home/hyunjoon/github/model_mxnet/mobilenet/mobilenet \
    --epoch 0 \
    --end-epoch 130 \
    --frequent 100 \
    --nms 0.45 \
    --gpus 6,7
    # --wd 1e-04 \
    # --lr-steps 3,3,4 \
    # --lr-factor 0.1 \
    # --optimizer-name sgd \
    # --use-plateau 1 \
    # --use-global-stats 1 \
