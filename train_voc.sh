python train.py \
    --train-path ./data/VOCdevkit/train.rec \
    --val-path ./data/VOCdevkit/val.rec \
    --num-class 20 \
    --class-names ./dataset/names/pascal_voc.names \
    --network symbol_mobilenetv4_yolo \
    --label-width 350 \
    --batch-size 20 \
    --data-shape 448 \
    --min-random-shape 320 \
    --max-random-shape 576 \
    --random-shape-step 64 \
    --freeze '^(conv1|conv2).*' \
    --pretrained /home/hyunjoon/github/model_mxnet/mobilenetv4/mobilenetv4_imagenet \
    --epoch 109 \
    --lr 1e-03 \
    --lr-steps 100,150 \
    --lr-factor 0.1 \
    --end-epoch 300 \
    --frequent 100 \
    --nms 0.35 \
    --gpus 0,1
    # --wd 1e-04 \
    # --lr-steps 3,3,4 \
    # --lr-factor 0.1 \
    # --optimizer-name sgd \
    # --use-plateau 1 \
    # --use-global-stats 1 \
