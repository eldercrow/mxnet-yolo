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
    --optimizer-name gnadam \
    --freeze '^(conv1|conv2).*' \
    --lr 5e-04 \
    --wd 1e-04 \
    --lr-factor 0.88586679 \
    --lr-steps 4, \
    --end-epoch 160 \
    --frequent 100 \
    --resume 4 \
    --gpus 0,1
    # --pretrained ./pretrained/mobilenet/mobilenet \
    # --epoch 0 \
