# object_detection/yolo_v3_backbone_mltools/tflite_int8

## Description
Backbone of the Yolo v3 model with an input size of 416 x 416. The backbone is quantized with an int8 precision using the first 1000 images of the COCO 2014 training set for calibration. The DarkNet original pre-trained weights are used as initial weights.

## License
[MIT](https://spdx.org/licenses/MIT.html)
[MIT]https://github.com/zzh8829/yolov3-tf2/blob/master/LICENSE

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 4adc0b716c5af29d957396fab2bcbc460e8b94ee |
|  Size (Bytes)       | 62958128 |
|  Provenance         | https://confluence.arm.com/display/MLENG/Yolo+v3 |
|  Paper | https://pjreddie.com/media/files/papers/YOLOv3.pdf |


## Accuracy
Dataset: coco-val-2014

| Metric | Value |
|--------|-------|
| mAP50 | 0.563 |

## Network Inputs
| Input Node Name | Shape | Example Path | Example Type | Example Use Case |
|-----------------|-------|--------------|------------------|--------------|
| input_int8 | (1, 416, 416, 3) | models/object_detection/yolo_v3_backbone_mltools/tflite_int8/testing_input/0.npy | int8 |  |

## Network Outputs
| Output Node Name | Shape | Description |
|------------------|-------|-------------|
| Identity_int8 | (1, 13, 13, 3, 85) | None |
| Identity_1_int8 | (1, 26, 26, 3, 85) | None |
| Identity_2_int8 | (1, 52, 52, 3, 85) | None |

