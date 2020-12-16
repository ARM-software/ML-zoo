# DS-CNN Large INT8

## Description
This is a fully quantized version (asymmetrical int8) of the DS-CNN Large model developed by Arm, with training checkpoints, from the Hello Edge paper. Code to recreate this model can be found here: https://github.com/ARM-software/ML-examples/tree/master/tflu-kws-cortex-m

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | abaa9d4bf8797801276c00151ee14426aa1b2dcc |
|  Size (Bytes)       | 530688 |
|  Provenance         | https://github.com/ARM-software/ML-examples/tree/master/tflu-kws-cortex-m |
|  Paper              | https://arxiv.org/abs/1711.07128 |

## Accuracy
Dataset: Google Speech Commands Test Set

| Metric | Value |
|--------|-------|
| Accuracy | 94.58% |

## Performance
| Platform | Optimized |
| -------- | ---------- |
|   CPU    |      :heavy_check_mark:      |
|   GPU    |      :heavy_check_mark:      |

### Key
 - :heavy_check_mark: - Optimized for the platform.
 - :heavy_minus_sign: - Not optimized, but will run on the platform.
 - :heavy_multiplication_x: - Not optimized and will not run on the platform.

## Optimizations
| Optimization |  Value  |
|-----------------|---------|
| Quantization | INT8 |

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| input | (1, 490) | The input is a processed MFCCs of shape (1, 490) |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| Identity | (1, 12) | The probability on 12 keywords. |
