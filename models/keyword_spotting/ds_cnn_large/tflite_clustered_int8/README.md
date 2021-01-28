# DS-CNN Clustered INT8

## Description
This is a clustered (32 clusters, kmeans++ centroid initialization), retrained (fine-tuned) and fully quantized version (INT8) of the DS-CNN Large model developed by Arm from the Hello Edge paper. Code for the original DS-CNN implementation can be found here: https://github.com/ARM-software/ML-KWS-for-MCU. 
The original model was converted to Keras, optimized using the Clustering API in TensorFlow Model Optimization Toolkit, and quantized using post-training quantization in the TF Lite Converter.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 367391990618e4e9a821a847dcbf3fe129212282 |
|  Size (Bytes)       | 526192 |
|  Provenance         | The original model (before clustering) is a pretrained checkpoint based on https://github.com/ARM-software/ML-KWS-for-MCU |
|  Paper              | https://arxiv.org/abs/1711.07128 |

## Accuracy
Dataset: Google Speech Commands

| Metric | Value |
|--------|-------|
| Top 1 Accuracy | 0.94701 |

## Optimizations
| Optimization |  Value  |
|-----------------|---------|
| Quantization | INT8 |
| Number of Clusters | 32 |
| Cluster Initialization | K-Means |

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| input_2 | (1, 1, 49, 10) | The input is a processed MFCCs of shape (1,1,49,10) |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| Identity | (1, 12) | The probability on 12 keywords. |
