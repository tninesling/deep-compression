# Deep Compression

UIUC CS521, Research Project Exploring Deep Compression for DNNs

# Overview

Deep Compression is a pipeline for reducing the size of deep neural nets using a combination of pruning, quantization, and encoding, originally described by [Han, Mao, and Dally](https://arxiv.org/pdf/1510.00149.pdf).

# Goals

Reproduce the paper's results, using the PyTorch framework

- [ ] For AlexNet
- [ ] For VGG-16

Explore the following pruning techniques

- [ ] L1 structured pruning
- TBD

Explore the following quantization techniques

- [ ] Incremental network quantization
- TBD

# Development

## ImageNet

The `alexnet.py` script expects that the ImageNet dataset hosted on [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) is available locally. The annotations and data are in separate directories, which means they'll need to be zipped together for validating the model. The data loader expects the locations of those two directories are available in the environment. They can be populated in [.env](.env) as `IMAGENET_ANNOTATIONS_DIR` and `IMAGENET_DATA_DIR`, respectively.
