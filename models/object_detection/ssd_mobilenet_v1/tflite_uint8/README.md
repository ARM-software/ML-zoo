# SSD MobileNet v1 UINT8

## Description
SSD MobileNet v1 is a object detection network, that localizes and identifies objects in an input image. This is a TF Lite quantized version that takes a 300x300 input image and outputs detections for this image. This model is trained and quantized by Google.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

### Model Recreation Code
Code to recreate this model can be found [here](recreate_model/).

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 1f9c945db9e32c33e5b91539f756a8fbef636405 |
|  Size (Bytes)       | 6898880 |
|  Provenance         | http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz |
|  Paper              | https://arxiv.org/abs/1512.02325 |

## Accuracy
Dataset: Coco Validation 2017

| Metric | Value |
|--------|-------|
| mAP | 0.180 |

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_multiplication_x:         |
| Cortex-M |:heavy_multiplication_x:         |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_multiplication_x:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.



## Optimizations
| Optimization |  Value  |
|-----------------|---------|
| Quantization | UINT8 |

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| normalized_input_image_tensor | (1, 300, 300, 3) | Input RGB images (a range of 0-255 per RGB channel). |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| TFLite_Detection_PostProcess | () | The y1, x1, y2, x2 coordinates of the bounding boxes for each detection |
| TFLite_Detection_PostProcess:1 | () | The class of each detection |
| TFLite_Detection_PostProcess:2 | () | The probability score for each classification |
| TFLite_Detection_PostProcess:3 | () | A vector containing a number corresponding to the number of detections |
