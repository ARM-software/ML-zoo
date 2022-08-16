# object_detection/yolo_v3_tiny/tflite_pruned_backbone_only_int8

## Description
YOLO v3 Tiny is the light version of YOLO v3 with less layers for object detection and classification.  
  This model contains only the backbone, and using Darknet pre-trained weights.

## License
[MIT License](https://github.com/zzh8829/yolov3-tf2/blob/master/LICENSE)

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | ec4c5ad5c92fe6bb7eb750011b0b1e322a15ba19 |
|  Size (Bytes)       | 8963352 |
|  Provenance         | https://github.com/zzh8829/yolov3-tf2 + https://pjreddie.com/media/files/yolov3-tiny.weights |
|  Paper | https://arxiv.org/pdf/1804.02767.pdf |


## DataSet
| Dataset Information | Value |
|--------|-------|
| Name | Microsoft Coco 2014 |
| Description | COCO is a large-scale object detection, segmentation, and captioning dataset. |
| Link | https://cocodataset.org/#home |


## Accuracy

| Metric | Value |
|--------|-------|
| mAP | 0.345 |

## Network Inputs
| Input Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| serving_default_input:0 | (1, 416, 416, 3) | int8 | models/object_detection/yolo_v3_tiny/tflite_pruned_backbone_only_int8/testing_input/serving_default_input:0 | int8 | [1, 416, 416, 3] | Random input for model regression. |

## Network Outputs
| Output Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| StatefulPartitionedCall:0 | (1, 13, 13, 3, 85) | int8 | models/object_detection/yolo_v3_tiny/tflite_pruned_backbone_only_int8/testing_output/StatefulPartitionedCall:0 | int8 | [1, 13, 13, 3, 85] | output for model regression. |
| StatefulPartitionedCall:1 | (1, 26, 26, 3, 85) | int8 | models/object_detection/yolo_v3_tiny/tflite_pruned_backbone_only_int8/testing_output/StatefulPartitionedCall:1 | int8 | [1, 26, 26, 3, 85] | output for model regression. |

