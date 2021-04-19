# DS-CNN Clustered FP32

## Description
This is a clustered (32 clusters, kmeans++ centroid initialization) and retrained (fine-tuned) FP32 version of the DS-CNN Large model developed by Arm from the Hello Edge paper. Code for the original DS-CNN implementation can be found here: https://github.com/ARM-software/ML-KWS-for-MCU. The original model was converted to Keras and optimized using the Clustering API in TensorFlow Model Optimization Toolkit.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

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

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:         |
| Cortex-M |:heavy_multiplication_x:         |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_multiplication_x:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.



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
