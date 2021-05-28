# ML-Team-Research

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![TensorFlowJS 3.3](https://img.shields.io/badge/TensorFlow.js-3.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v3.3.0)

## Installation

### Tensorflow Object Detection API with Tensorflow 2

Please refer to this [document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

### Tensorflow.js

```
pip install tensorflowjs
```

### Weights & Biases

We use [Weights & Biases](https://wandb.ai/site) to track the result of experiments.

```
pip install wandb
```

## Conversion

Convert trained **Tensorflow Checkpoint** to **TensorFlow.js JSON** format.

```
python converter_script.py \
    --pipeline_config_path path/to/pipeline.config \
    --trained_checkpoint_dir path/to/checkpoint \
    --output_directory path/to/coverted_output_directory
```

