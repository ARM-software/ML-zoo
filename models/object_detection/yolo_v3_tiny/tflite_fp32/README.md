# YOLO v3 Tiny FP32

## Description
Yolo v3 Tiny is a object detection network, that localizes and identifies objects in an input image. This is a floating point version that takes a 416x416 input image and outputs detections for this image. This model is generated using the weights from the [https://pjreddie.com/darknet/yolo/](YOLO website).

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

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
| -------- | ---------- |
|   CPU    |      :heavy_check_mark:      |
|   GPU    |      :heavy_check_mark:      |

### Key
 - :heavy_check_mark: - Optimized for the platform.
 - :heavy_minus_sign: - Not optimized, but will run on the platform.
 - :heavy_multiplication_x: - Not optimized and will not run on the platform.

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| inputs | (1, 416, 416, 3) | A 416x416 floating point input image. |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| output_boxes | (1, 2535, 85) | A 1xNx85 map of predictions, where the first 4 entries of the 3rd dimension are the bounding box coordinates and the 5th is the confidence. The remaining entries are softmax scores for each class. |
