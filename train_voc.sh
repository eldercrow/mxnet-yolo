python train.py \
    --train-path ./data/VOCdevkit/train.rec \
    --val-path ./data/VOCdevkit/val.rec \
    --num-class 20 \
    --class-names ./dataset/names/pascal_voc.names \
    --network symbol_mobilenet_yolo \
    --label-width 350 \
    --batch-size 32 \
    --data-shape 384 \
    --freeze '^(conv1|conv2).*' \
    --pretrained /home/hyunjoon/github/model_mxnet/mobilenetv1/mobilenetv1 \
    --epoch 0 \
    --lr 1e-02 \
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
