# YOLO-v2: Real-Time Object Detection

Still under development. 71 mAP on VOC2007 achieved so far.

This is a pre-released version.

### Disclaimer
This is a re-implementation of original yolo v2 which is based on [darknet](https://github.com/pjreddie/darknet).
The arXiv paper is available [here](https://arxiv.org/pdf/1612.08242.pdf).

#### (eldercrow)
I am not interested in exact re-implementation of the original YOLO.
If anyone is looking for that, check zhreshold's original repo.

This code is now sorta working (training only), but the mAP is low yet.
You don't need to complie the custom mxnet to use this branch.

Changes from the original algorithm:
- [Focal loss](https://arxiv.org/abs/1708.02002) for classifier training.
- [IOU loss](https://arxiv.org/abs/1608.01471) for regressor training.
- MobileNet for backbone network.

TODO:
- Apply plateau learning scheduler.
- (done) Replace cpp layers to python layers.
- (done) Add bounding box clutering method.
- (done) Apply focal loss.
- (done) Apply IOU loss for regressor training.

### Demo

![demo1](https://user-images.githubusercontent.com/3307514/28980832-29bb0262-7904-11e7-83e3-a5fec65e0c70.png)

### Getting started
- ~~Build from source, this is required because this example is not merged, some
custom operators are not presented in official MXNet. [Instructions](http://mxnet.io/get_started/install.html)~~
- Install required packages: `cv2`, `matplotlib`

### Try the demo
- Download the pretrained [model](https://github.com/zhreshold/mxnet-yolo/releases/download/0.1-alpha/yolo2_darknet19_416_pascalvoc0712_trainval.zip), and extract to `model/` directory.
- Pretrained model for mobilenet can be found [here](https://github.com/KeyKy/mobilenet-mxnet).
- Run
```
# cd /paht/to/mxnet-yolo
python demo.py --cpu
# available options
python demo.py -h
```

### Train the model
- Grab a pretrained model, e.g. [`darknet19`](https://github.com/zhreshold/mxnet-yolo/releases/download/0.1-alpha/darknet19_416_ILSVRC2012.zip)
- Download PASCAL VOC dataset.
```
cd /path/to/where_you_store_datasets/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
ln -s /path/to/VOCdevkit /path/to/mxnet-yolo/data/VOCdevkit
```
- Create packed binary file for faster training
```
# cd /path/to/mxnet-ssd
bash tools/prepare_pascal.sh
# or if you are using windows
python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target ./data/train.lst
python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target ./data/val.lst --shuffle False
```
- Start training
```
python train.py --gpus 0,1,2,3 --epoch 0
```
