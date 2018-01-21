python demo.py \
    --network mobilenet_yolo \
    --images ./data/demo/street.jpg \
    --prefix ./model/yolo2_mobilenet_576 \
    --epoch 117 \
    --data-shape 576 \
    --thresh 0.5 \
    --nms 0.45 \
    --cpu

