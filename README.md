# DeepSelfie: Single-shot Low-light Enhancement for Selfies

If you are interested in using our code and find it useful in your research, please consider citing the following paper:

```latex
@article{article,
author = {Lu, Yucheng and Kim, Dong-Wook and Jung, Seung-Won},
year = {2020},
month = {07},
pages = {1-1},
title = {DeepSelfie: Single-shot Low-light Enhancement for Selfies},
volume = {PP},
journal = {IEEE Access},
doi = {10.1109/ACCESS.2020.3006525}
}
```



### System Requirements

- python 3
- pyTorch
- torchvision
- Jupyter Notebook (for training)
- Visdom (for training)
- OpenCV (for training)
- numpy
- build-essential: sudo apt-get install build-essential
- python-all-dev: sudo apt-get install python-all-dev
- libexiv2-dev: sudo apt-get install libexiv2-dev
- libboost-python-dev: sudo apt-get install libboost-python-dev
- pyexiv2: pip install py3exiv2
- libraw: sudo apt-get install libraw-dev
- rawpy: pip install rawpy

Please note that the libraw installed via apt-get is an outdated version, if you get wrong results caused by this version, you should better build it from [source](https://github.com/LibRaw/LibRaw).



### Run Demo

To run demo, download the pretrained models from [onedrive](https://dongguk0-my.sharepoint.com/:f:/g/personal/yc_lu_dongguk_edu/EoIwKeaFgZhAj9UaLoxVWDEBX3Yhs07mpXyJn5Y_Xj6aTQ?e=30kDcX), unzip and copy the files to "saves" folder, add your .DNG files to "samples" folder, then run the following command:

`python ./demo.py`

The generated images will be saved to "results" folder.

Alternatively, you can specify your own input or output path by passing "--input" and "--output":

`python ./demo.py --input PATH-TO-INPUT-FOLDER --output PATH-TO-OUTPUT-FOLDER`

The default device used for running this demo is CPU, if you want to use GPU instead, pass "--device cuda":

`python ./demo.py --device cuda`

If you encounter out of memory error, try down-sampling the input before further processing:

`python ./demo.py --resize (600,800)`

The output images are not calibrated and thus have distortion, if you want to do camera calibration, pass "--calib":

`python ./demo.py --calib`



### Train from scratch

Before training from scratch, you need to first download the FivekNight dataset from [onedrive](https://dongguk0-my.sharepoint.com/:f:/g/personal/yc_lu_dongguk_edu/EoIwKeaFgZhAj9UaLoxVWDEBX3Yhs07mpXyJn5Y_Xj6aTQ?e=30kDcX). Also, you need to use your camera to take some images (the more the better, with various exposure levels and ISO) in RAW format. To start training, follow the instructions below:

1. Generate training dataset for r2rNet by running "genDB-r2rNet.ipynb" under /dataset.
2. Get the statistical information of your own dataset by running "getStat_r2rNet.ipynb" under /dataset, copy the results to "config.ipynb" under /.
3. Train r2rNet by running "train-dataloader.ipynb" under /, you may want to specify a new port for Visdom.
4. Generate validation dataset by running "genValset.ipynb" under /, this step should be performed after r2rNet training is completed.
5. Train the gain estimation network by running "train-gainEst.ipynb" under /, you may want to specify a new port for Visdom.
6. Train the raw processing network by running "train-rawProcess.ipynb" under /, you may want to specify a new port for Visdom.

Note: You may need to change the paths at step 1 and 2, and the port used by Visdom at step 3, 5, and 6.

