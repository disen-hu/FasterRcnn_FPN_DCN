# FasterRcnn_FPN_DCN
* This is an official implementation for the paper [Object recognition in atmospheric turbulence scenes](https://arxiv.org/abs/2210.14318).
* This repo use a synthetic atmospheric turbulence dataset for trainning and test.


# Citation
If you think this repo is useful,please citing :
```
@article{hu22,Object recognition in atmospheric turbulence scenes
    Author = {Disen Hu,Nantheera Anantrasirichai},
    Title = {Object recognition in atmospheric turbulence scenes},
    Journal = {arXiv preprint arXiv:2210.14318},
    Year = {2022}
}
```

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

## Resource：

* dataset used：[OneDrive](https://1drv.ms/u/s!An4ptxH2n0OJbTkK3zLW4bjfnsc?e=ppRE3g) unzip this dataset in data/ 
* pre-trained weights: [OneDrive](https://1drv.ms/u/s!An4ptxH2n0OJc19Uj-AWM-hQ41g?e=KTtyqu) To use the pre-trained weights to train, plz please this weight file in the backbone/
* our-trained weights: [OneDrive](https://1drv.ms/u/s!An4ptxH2n0OJbt8Q_Q3z8dta0aE?e=kcKhtN) This is our trained weights, put it in backbone/ use command python validation.py to get mAP.


# To do: 
* 1.translate code comments from Chinese to English.
* 2.split work in different folder
* 
