# YOLO v3 Tiny FP32

## Description
Yolo v3 Tiny is a object detection network, that localizes and identifies objects in an input image. This is a floating point version that takes a 416x416 input image and outputs detections for this image. This model is generated using the weights from the [YOLO website](https://pjreddie.com/darknet/yolo/).

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

### Model Recreation Code
Code to recreate this model can be found [here](recreate_model/).

### How-To Guide
A guide on how to deploy this model using the Arm NN SDK can be found [here](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/object-recognition-with-arm-nn-and-raspberry-pi).

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | b38f7be6856eed4466493bdc86be1879f4b743fb |
|  Size (Bytes)       | 35455980 |
|  Provenance         | https://pjreddie.com/media/files/yolov3-tiny.weights & https://github.com/mystic123/tensorflow-yolo-v3 |
|  Paper              | https://arxiv.org/abs/1804.02767 |

## Accuracy
Dataset: MS COCO Validation

| Metric | Value |
|--------|-------|
| mAP | 0.331 |

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
| inputs | (1, 416, 416, 3) | A 416x416 floating point input image. |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| output_boxes | (1, 2535, 85) | A 1xNx85 map of predictions, where the first 4 entries of the 3rd dimension are the bounding box coordinates and the 5th is the confidence. The remaining entries are softmax scores for each class. |

## Sources
- [DarkNet](https://github.com/pjreddie/darknet/blob/master/LICENSE)
- [YOLO v3 Paper](https://arxiv.org/abs/1804.02767)
