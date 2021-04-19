
# Model Zoo 
![version](https://img.shields.io/badge/version-20.12-0091BD)
> A collection of machine learning models optimized for Arm IP.


## Image Classification

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="100">Cortex-A</th>
        <th width="100">Cortex-M</th>
        <th width="100">Mali GPU</th>
        <th width="100">Ethos U</th>
    </tr>
    <tr>
        <td>[MobileNet v2 1.0 224 UINT8](models/image_classification/mobilenet_v2_1.0_224/tflite_uint8)</td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
</table>

## Keyword Spotting

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="100">Cortex-A</th>
        <th width="100">Cortex-M</th>
        <th width="100">Mali GPU</th>
        <th width="100">Ethos U</th>
    </tr>
    <tr>
        <td>[CNN Large INT8 *](models/keyword_spotting/cnn_large/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[CNN Medium INT8 *](models/keyword_spotting/cnn_medium/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[CNN Small INT8 *](models/keyword_spotting/cnn_small/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DNN Large INT8 *](models/keyword_spotting/dnn_large/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DNN Medium INT8 *](models/keyword_spotting/dnn_medium/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DNN Small INT8 *](models/keyword_spotting/dnn_small/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DS-CNN Clustered FP32](models/keyword_spotting/ds_cnn_large/tflite_clustered_fp32)</td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
    <tr>
        <td>[DS-CNN Clustered INT8](models/keyword_spotting/ds_cnn_large/tflite_clustered_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DS-CNN Large INT8 *](models/keyword_spotting/ds_cnn_large/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DS-CNN Medium INT8 *](models/keyword_spotting/ds_cnn_medium/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>[DS-CNN Small INT8 *](models/keyword_spotting/ds_cnn_small/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
</table>

## Object Detection

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="100">Cortex-A</th>
        <th width="100">Cortex-M</th>
        <th width="100">Mali GPU</th>
        <th width="100">Ethos U</th>
    </tr>
    <tr>
        <td>[SSD MobileNet v1 FP32 *](models/object_detection/ssd_mobilenet_v1/tflite_fp32)</td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
    <tr>
        <td>[SSD MobileNet v1 UINT8 *](models/object_detection/ssd_mobilenet_v1/tflite_uint8)</td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
    <tr>
        <td>[YOLO v3 Tiny FP32 *](models/object_detection/yolo_v3_tiny/tflite_fp32)</td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
</table>

## Speech Recognition

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="100">Cortex-A</th>
        <th width="100">Cortex-M</th>
        <th width="100">Mali GPU</th>
        <th width="100">Ethos U</th>
    </tr>
    <tr>
        <td>[Wav2letter INT8](models/speech_recognition/wav2letter/tflite_int8)</td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
</table>

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.
* `*` - Code to recreate model available.


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) unless otherwise explicitly stated.
