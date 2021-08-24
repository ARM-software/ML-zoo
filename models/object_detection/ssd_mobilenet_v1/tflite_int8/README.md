# SSD MobileNet v1 INT8

## Description
SSD MobileNet v1 is a object detection network, that localizes and identifies objects in an input image. This is a TF Lite quantized version that takes a 300x300 input image and outputs detections for this image. This model is converted from FP32 to INT8 using post-training quantization.

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
|  SHA-1 Hash         | fef68428bd439b70eb983b57d6a342871fa0deaa |
|  Size (Bytes)       | 7311392 |
|  Provenance         | https://arxiv.org/abs/1512.02325 |
|  Paper              | https://arxiv.org/abs/1512.02325 |

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

## Accuracy
Dataset: COCO 2017 Validation

| Metric | Value |
|--------|-------|
| mAP | 0.234 |

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
        <td>(1, 300, 300, 3)</td>
        <td>A resized and normalized input image.</td> 
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
        <td>TFLite_Detection_PostProcess:01</td>
        <td>()</td>
        <td>The y1, x1, y2, x2 coordinates of the bounding boxes for each detection</td> 
    </tr>
    <tr>
        <td>TFLite_Detection_PostProcess:02</td>
        <td>()</td>
        <td>The class of each detection</td> 
    </tr>
    <tr>
        <td>TFLite_Detection_PostProcess:03</td>
        <td>()</td>
        <td>The probability score for each classification</td> 
    </tr>
    <tr>
        <td>TFLite_Detection_PostProcess:04</td>
        <td>()</td>
        <td>A vector containing a number corresponding to the number of detections</td> 
    </tr>
</table>
