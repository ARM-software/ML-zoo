# MobileNet v2 1.0 224 INT8

## Description
INT8 quantised version of MobileNet v2 model. Trained on ImageNet.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

### Model Recreation Code
Code to recreate this model can be found [here](recreate_model/).

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 8de7996dfeadb5ab6f09e3114f3905fd03879eee |
|  Size (Bytes)       | 4020936 |
|  Provenance         | https://arxiv.org/pdf/1801.04381.pdf |
|  Paper              | https://arxiv.org/pdf/1801.04381.pdf |

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
Dataset: ILSVRC 2012

| Metric | Value |
|--------|-------|
| Top 1 Accuracy | 0.697 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |

## Network Inputs
<table>
    <tr>
        <th width="200">Input Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>tfl.quantize</td>
        <td>(1, 224, 224, 3)</td>
        <td>Single 224x224 RGB image with INT8 values between -128 and 127</td>
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
        <td>MobilenetV2/Predictions/Reshape_11</td>
        <td>(1, 1001)</td>
        <td>Per-class confidence for 1001 ImageNet classes</td>
    </tr>
</table>
