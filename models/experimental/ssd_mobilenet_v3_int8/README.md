# ssd_mobilenet_v3_int8.tflite

## Description
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2019_08_14.tar.gz
This model is directly derived from the above URLs, with only the post processing removed

## License
https://github.com/tensorflow/models/blob/master/LICENSE
Apache v2

## Network Inputs
| Input Node Name | Shape | Example Path | Example Type | Example Use Case |
|-----------------|-------|--------------|------------------|--------------|
| normalized_input_image_tensor | (1, 320, 320, 3) | N/A | N/A | N/A |

## Network Outputs
| Output Node Name | Shape | Description |
|------------------|-------|-------------|
| raw_outputs/class_predictions | (1, 2034, 91) | Class predictions |
| raw_outputs/box_encodings | (1, 2034, 4) | Boxe Encodings |
