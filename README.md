# Model Zoo 
![version](https://img.shields.io/badge/version-20.12-0091BD)
> A collection of machine learning models optimized for Arm IP.

## Anomaly Detection

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
        <td><a href = "models/anomaly_detection/micronet_large/tflite_int8">MicroNet Large INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href = "models/anomaly_detection/micronet_medium/tflite_int8">MicroNet Medium INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href = "models/anomaly_detection/micronet_small/tflite_int8">MicroNet Small INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
</table>

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
        <td><a href="models/image_classification/mobilenet_v2_1.0_224/tflite_uint8">MobileNet v2 1.0 224 UINT8</a></td>
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
        <td><a href="models/keyword_spotting/cnn_large/tflite_int8">CNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_medium/tflite_int8">CNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_small/tflite_int8">CNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_large/tflite_int8">DNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_medium/tflite_int8">DNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_small/tflite_int8">DNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/tflite_clustered_fp32">DS-CNN Clustered FP32</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/tflite_clustered_int8">DS-CNN Clustered INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/tflite_int8">DS-CNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_medium/tflite_int8">DS-CNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_small/tflite_int8">DS-CNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href = "models/keyword_spotting/micronet_large/tflite_int8">MicroNet Large INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href = "models/keyword_spotting/micronet_medium/tflite_int8">MicroNet Medium INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href = "models/keyword_spotting/micronet_small/tflite_int8">MicroNet Small INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
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
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_fp32">SSD MobileNet v1 FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_uint8">SSD MobileNet v1 UINT8 *</a></td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/yolo_v3_tiny/tflite_fp32">YOLO v3 Tiny FP32 *</a></td>
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
        <td><a href="models/speech_recognition/wav2letter/tflite_pruned_int8">Wav2letter Pruned INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/wav2letter/tflite_int8">Wav2letter INT8</a></td>
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