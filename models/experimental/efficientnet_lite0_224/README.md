# image_classification/efficientnet_lite0_224/tflite_int8

## Description
This work is developed from the codebase located [here](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md) and is under an Apache 2 license available [here](https://github.com/tensorflow/tpu/blob/master/LICENSE). 

The original networks, which we have optimized via tooling but left otherwise unchanged are copyright the tensorflow authors as in the license file linked.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 35f9dafaf25f8abf2225265b0724979a68bf6d67 |
|  Size (Bytes)       | 5422760 |
|  Provenance         | https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz |
|  Paper | https://arxiv.org/pdf/1905.11946.pdf |


## Accuracy
Dataset: ILSVRC 2012

| Metric | Value |
|--------|-------|
| top_1_accuracy | 0.744 |

## Network Inputs
| Input Node Name | Shape | Example Path | Example Type | Example Use Case |
|-----------------|-------|--------------|------------------|--------------|
| images | (1, 224, 224, 3) | models/image_classification/efficientnet_lite0_224/tflite_int8/testing_input |  | Typical ImageNet-style single-batch cat resized to 224x224. |

## Network Outputs
| Output Node Name | Shape | Description |
|------------------|-------|-------------|
| Softmax | (1, 1000) | Probability distribution over 1000 ImageNet classes with uint8 values. |

