# SSD MobileNet v1 FP32

## Description
SSD MobileNet v1 is a object detection network, that localizes and identifies objects in an input image. This is a TF Lite floating point version that takes a 300x300 input image and outputs detections for this image. This model is trained by Google.

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
|  SHA-1 Hash         | 5bd511fc17ec7bfe9cd0f51bdec1537b874f52d2 |
|  Size (Bytes)       | 27286108 |
|  Provenance         | http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz |
|  Paper              | https://arxiv.org/abs/1512.02325 |

## Accuracy
Dataset: Coco Validation 2017

| Metric | Value |
|--------|-------|
| mAP | 0.21 |

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



## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| normalized_input_image_tensor | (1, 300, 300, 3) | A float input image. |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| TFLite_Detection_PostProcess | () | An array of num_detection box boundaries for each input in the format (y1, x1, y2, x2) scaled from 0 to 1. |
| TFLite_Detection_PostProcess:1 | () | COCO detection classes for each object. 1=person, 11=fire hydrant. |
| TFLite_Detection_PostProcess:2 | () | Detection scores for each object. |
| TFLite_Detection_PostProcess:3 | () | The number of objects detected in each image. |
