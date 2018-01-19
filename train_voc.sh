python train_imdb.py \
    --network mobilenet_yolo \
    --dataset pascal_voc \
    --devkit-path ./data/VOCdevkit \
    --year 2007,2012 \
    --image-set trainval \
    --val-image-set test \
    --val-year 2007 \
    --batch-size 32 \
    --data-shape 576 \
    --optimizer-name sgd \
    --freeze '^(conv1|conv2).*' \
    --lr 5e-03 \
    --wd 1e-04 \
    --lr-factor 0.1 \
    --lr-steps 80,120, \
    --end-epoch 160 \
    --frequent 100 \
    --pretrained ./model/yolo2_mobilenet_576 \
    --epoch 96 \
    --gpus 0,1
    # --pretrained ./pretrained/mobilenet/mobilenet \
    # --epoch 0 \
