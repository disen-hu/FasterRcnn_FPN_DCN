# FasterRcnn_FPN_DCN
An implementation of FasterRcnn_FPN_DCN.



To do: 
1.translate code comments from Chinese to English.
2.split work in different folder
3.upload dataset and trained weights.


# Faster R-CNN

## This repo is implementated by touchvision'repo
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

## Excuting enviroment：
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`)
* Ubuntu or Centos
* Use GPU to train
* All detail enviroment requirements in `requirements.txt`

## File structure：
```
  ├── backbone: Feature extraction network 
  ├── network_files: Faster R-CNN
  ├── train_utils: Training and validation related modules (including cocotools)
  ├── my_dataset.py: load dataset
  ├── train_resnet50_fpn.py: resnet50+FPN as backbone to train
  ├── train_multi_GPU.py: for multi GPU training
  ├── predict.py: script to get predict results
  ├── validation.py: get map from trained weights and datasets
  └── pascal_voc_classes.json: label of pascal_voc
```


 
 