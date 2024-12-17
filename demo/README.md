## Getting Started with DI-MaskDINO

This document provides a brief intro of the usage of **DI-MaskDINO**.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Inference Demo with Pre-trained Models

1. Pick a model and its config file
- for example
   - config file at `configs/dimaskdino_r50_4scale_bs16_12ep.yaml`.
   - Model file [DI-MaskDINO (12ep) ](./configs/dimaskdino_r50_4scale_bs16_12ep.yaml)
2. We provide `demo.py` that is able to demo builtin configs. 
3. Run it with:
```
cd demo/
python demo.py --config-file configs/dimaskdino_r50_4scale_bs16_12ep.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/model_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


