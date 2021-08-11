# DS-CNN Clustered INT8

## Description
This is a clustered (32 clusters, kmeans++ centroid initialization), retrained (fine-tuned) and fully quantized version (INT8) of the DS-CNN Large model developed by Arm from the Hello Edge paper. Code for the original DS-CNN implementation can be found here: https://github.com/ARM-software/ML-KWS-for-MCU. The original model was converted to Keras, optimized using the Clustering API in TensorFlow Model Optimization Toolkit, and quantized using post-training quantization in the TF Lite Converter.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

### Model Recreation Code
Code to recreate this model can be found here: https://github.com/ARM-software/ML-examples/tree/master/tflu-kws-cortex-m.

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 2ee38794ed171c75d3313460a1633c5d6a79f530 |
|  Size (Bytes)       | 503816 |
|  Provenance         | The original model (before clustering) is a pretrained checkpoint based on https://github.com/ARM-software/ML-KWS-for-MCU |
|  Paper              | https://arxiv.org/abs/1711.07128 |

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:         |
| Cortex-M |:heavy_check_mark:         |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_check_mark:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Accuracy
Dataset: Google Speech Commands Test Set

| Metric | Value |
|--------|-------|
| Top 1 Accuracy | 0.9401 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |
| Number of Clusters | 32 |
| Cluster Initialization | K-Means |

## Network Inputs
<table>
    <tr>
        <th width="200">Input Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>input</td>
        <td>(1, 490)</td>
        <td>The input is a processed MFCCs of shape (1,490)</td> 
    </tr>
</table>

## Network Outputs
<table>
    <tr>
        <th width="200">Output Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>Identity</td>
        <td>(1, 12)</td>
        <td>The probability on 12 keywords.</td> 
    </tr>
</table>
