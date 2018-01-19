python demo.py \
    --network mobilenet_yolo \
    --images ./data/demo/eagle.jpg \
    --prefix ./model/yolo2_mobilenet_576 \
    --epoch 95 \
    --data-shape 576 \
    --thresh 0.5 \
    --cpu

