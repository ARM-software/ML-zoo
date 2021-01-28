# DS-CNN Clustered FP32

## Description
This is a clustered (32 clusters, kmeans++ centroid initialization) and retrained (fine-tuned) FP32 version of the DS-CNN Large model developed by Arm from the Hello Edge paper. Code for the original DS-CNN implementation can be found here: https://github.com/ARM-software/ML-KWS-for-MCU. The original model was converted to Keras and optimized using the Clustering API in TensorFlow Model Optimization Toolkit.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | a6f4fa3e253125499a1e77aa1010c943e72af568 |
|  Size (Bytes)       | 1649816 |
|  Provenance         | The original model (before clustering and quantization) is a pretrained checkpoint based on https://github.com/ARM-software/ML-KWS-for-MCU |
|  Paper              | https://arxiv.org/abs/1711.07128 |

## Accuracy
Dataset: Google Speech Commands

| Metric | Value |
|--------|-------|
| Top 1 Accuracy | 0.9506 |

## Optimizations
| Optimization |  Value  |
|-----------------|---------|
| Number of Clusters | 32 |
| Cluster Initialization | K-Means |

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| input_4 | (1, 1, 49, 10) | The input is a processed MFCCs of shape (1,1,49,10) |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| Identity | (1, 12) | The probability on 12 keywords. |
