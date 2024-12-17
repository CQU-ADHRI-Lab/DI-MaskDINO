<h2 align="center"> <a href="https://arxiv.org/abs/2410.16707">DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model</a></h2>
<h4 align="center" color="A0A0A0"> Zhixiong Nan, Xianghong Li, Tao Xiang*, Jifeng Dai</h4>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2410.16707-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.16707)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO/blob/main/LICENSE)

</div>

<div align=center>
<img src="figures/results.gif" width="960px">
</div>



# DI-MaskDINO
This is the official implementation of the paper "DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model".

<div align="center">
  <img src="figures/framework.png"/>
</div><br/>

Motivated by an interesting phenomenon that _the performance of object detection lags behind that of instance segmentation at the beignning decoder layer_, **DI-MaskDINO** is proposed to alleviate the detection-segmentation imbalance issue.
**DI-MaskDINO** is implemented by configuring proposed **_De-Imbalance (DI)_** module and **_Balance-Aware Tokens Optimization (BATO)_** module to MaskDINO.
**_DI_** module contributes to strengthen the detection task at the beginning decoder layer to balance the performance of the two tasks, and the core of **_DI_** module is _residual double-selection mechanism_.
**DI-MaskDINO** outperforms existing SOTA joint object detection and instance segmentation model MaskDINO (**+1.2** _AP<sup>box</sup>_ and **+0.9** _AP<sup>mask</sup>_ on COCO, using ResNet50 backbone with 12 training epochs), SOTA object detection model DINO (**+1.0** _AP<sup>box</sup>_ on COCO), and SOTA segmentation model Mask2Former(**+3.0** _AP<sup>mask</sup>_ on COCO).

## Update
[2024/12] Code for [DI-MaskDINO](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO) is available here!

[2024/9] DI-MaskDINO has been accepted at NeurIPS 2024 as a poster!

## Installation

We tested our code with `Python=3.7.16, PyTorch=1.9.0, CUDA=11.1`. Please install PyTorch first according to [official instructions](https://pytorch.org/get-started/previous-versions/). Our code is based on [detectron2](https://github.com/facebookresearch/detectron2). Please refer to the [installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) of detectron2.

Example conda environment setup：

```bash
# Create a new virtual environment
conda create -n dimaskdino python=3.7
conda activate dimaskdino

# Install PyTorch
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

# Install opencv
pip install opencv-python

# Install detectron2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Under your working directory
git clone https://github.com/CQU-ADHRI-Lab/DI-MaskDINO.git
cd DI-MaskDINO
pip install -r requirements.txt

# CUDA kernel for MSDeformAttn
cd dimaskdino/modeling/pixel_decoder/ops
sh make.sh
```

## Models

We provide the checkpoint of the following models:

| Name                                                                                     | Backbone  | Epochs | **_AP<sup>box</sup>_** | **_AP<sup>mask</sup>_** | Download  |
|------------------------------------------------------------------------------------------| -------- | ------ |----------------------|-----------------------| --------- |
| [DI-MaskDINO](./configs/coco/instance-segmentation/dimaskdino_r50_4scale_bs16_12ep.yaml) | R50     | 12     | 46.9                 | 42.3                  | [model](https://drive.google.com/file/d/1oQociMQSt_jmQtH3pm92i6zat8Gt83Oh/view?usp=drive_link) |
| [DI-MaskDINO](./configs/coco/instance-segmentation/dimaskdino_r50_4scale_bs16_24ep.yaml) | R50     | 24     | 49.6                 | 44.8                  | [model](https://drive.google.com/file/d/1FjoYiwGnrk_bk0nUM7GPeIMGENVLdx7n/view?usp=drive_link) |
| [DI-MaskDINO](./configs/coco/instance-segmentation/dimaskdino_r50_4scale_bs16_50ep.yaml) | R50      | 50     | 51.9                 | 46.7                  | [model](https://drive.google.com/file/d/17_SErWYBWWCdYfR6i7OJtcBIlCDES10z/view?usp=drive_link) |

## Run

### Training

Train DI-MaskDINO with 8 GPUs:

```sh
python train_net.py --num-gpus 8 --config-file configs/coco/instance-segmentation/dimaskdino_r50_4scale_bs16_12ep.yaml OUTPUT_DIR /path/to/output
```

### Evaluation

You can download our pretrained models and evaluate them with the following commands. 
```sh
python train_net.py --eval-only --num-gpus 8 --config-file config_path MODEL.WEIGHTS /path/to/checkpoint_file
```
For example, to reproduce our result, you can copy the config path from the table, download the pretrained checkpoint into `/path/to/checkpoint_file`, and run 
```sh
python train_net.py --eval-only --num-gpus 8 --config-file configs/coco/instance-segmentation/dimaskdino_r50_4scale_bs16_12ep.yaml MODEL.WEIGHTS /path/to/checkpoint_file
```


## <a name="CitingDIMaskDINO"></a>Citing DI-MaskDINO

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@inproceedings{nan2024di,
  title={DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model}, 
  author={Zhixiong Nan and Xianghong Li and Tao Xiang and Jifeng Dai},
  booktitle={Proceedings of the Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgement

Many thanks to these excellent opensource projects 
* [MaskDINO](https://github.com/IDEA-Research/MaskDINO)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former) 
* [DINO](https://github.com/IDEA-Research/DINO)