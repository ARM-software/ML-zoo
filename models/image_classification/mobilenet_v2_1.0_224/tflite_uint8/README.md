# MobileNet v2 1.0 224 UINT8

## Description
MobileNet v2 is an efficient image classification neural network, targeted for mobile and embedded use cases. This model is trained on the ImageNet dataset and is quantized to the UINT8 datatype by Google.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 275c9649cb395139103b6d15f53011b1b949ad00 |
|  Size (Bytes)       | 3577760 |
|  Provenance         | https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224_quantized/1/default/1 |
|  Paper              | https://arxiv.org/pdf/1801.04381.pdf |

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:         |
| Cortex-M |:heavy_multiplication_x:         |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_check_mark:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.


## Accuracy
Dataset: Ilsvrc 2012

| Metric | Value |
|--------|-------|
| Top 1 Accuracy | 0.708 |

## Optimizations
| Optimization |  Value  |
|-----------------|---------|
| Quantization | UINT8 |

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| input | (1, 224, 224, 3) | Single 224x224 RGB image with UINT8 values between 0 and 255 |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| output | (1, 1001) | Per-class confidence for 1001 ImageNet classes |
