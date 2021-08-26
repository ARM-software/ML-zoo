
# Model Zoo
![version](https://img.shields.io/badge/version-21.08-0091BD)
> A collection of machine learning models optimized for Arm IP.


## Anomaly Detection

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="90">Cortex-A</th>
        <th width="90">Cortex-M</th>
        <th width="90">Mali GPU</th>
        <th width="90">Ethos U</th>
        <th width="90">Score (AUC)</th>
    </tr>
    <tr>
        <td><a href="models/anomaly_detection/micronet_large/tflite_int8">MicroNet Large INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.968</td>
    </tr>
    <tr>
        <td><a href="models/anomaly_detection/micronet_medium/tflite_int8">MicroNet Medium INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.963</td>
    </tr>
    <tr>
        <td><a href="models/anomaly_detection/micronet_small/tflite_int8">MicroNet Small INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.955</td>
    </tr>
</table>

**Dataset**: Dcase 2020 Task 2 Slide Rail

## Image Classification

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="90">Cortex-A</th>
        <th width="90">Cortex-M</th>
        <th width="90">Mali GPU</th>
        <th width="90">Ethos U</th>
        <th width="90">Score (Top 1 Accuracy)</th>
    </tr>
    <tr>
        <td><a href="models/image_classification/mobilenet_v2_1.0_224/tflite_int8">MobileNet v2 1.0 224 INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.697</td>
    </tr>
    <tr>
        <td><a href="models/image_classification/mobilenet_v2_1.0_224/tflite_uint8">MobileNet v2 1.0 224 UINT8</a></td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.708</td>
    </tr>
</table>

**Dataset**: ILSVRC 2012

## Keyword Spotting

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="90">Cortex-A</th>
        <th width="90">Cortex-M</th>
        <th width="90">Mali GPU</th>
        <th width="90">Ethos U</th>
        <th width="90">Score (Accuracy)</th>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_large/tflite_int8">CNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.929</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_medium/tflite_int8">CNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.913</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_small/tflite_int8">CNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.914</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_large/tflite_int8">DNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.863</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_medium/tflite_int8">DNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.846</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_small/tflite_int8">DNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.827</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/tflite_clustered_fp32">DS-CNN Clustered FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">0.950</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/tflite_clustered_int8">DS-CNN Clustered INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.940</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/tflite_int8">DS-CNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.946</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_medium/tflite_int8">DS-CNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.934</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_small/tflite_int8">DS-CNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.934</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/micronet_large/tflite_int8">MicroNet Large INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.965</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/micronet_medium/tflite_int8">MicroNet Medium INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.958</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/micronet_small/tflite_int8">MicroNet Small INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.953</td>
    </tr>
</table>

**Dataset**: Google Speech Commands Test Set

## Object Detection

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="90">Cortex-A</th>
        <th width="90">Cortex-M</th>
        <th width="90">Mali GPU</th>
        <th width="90">Ethos U</th>
        <th width="90">Score (mAP)</th>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_fp32">SSD MobileNet v1 FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">0.210</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_int8">SSD MobileNet v1 INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">0.234</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_uint8">SSD MobileNet v1 UINT8 *</a></td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">0.180</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/yolo_v3_tiny/tflite_fp32">YOLO v3 Tiny FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">0.331</td>
    </tr>
</table>

**Dataset**: COCO Validation 2017

## Speech Recognition

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="90">Cortex-A</th>
        <th width="90">Cortex-M</th>
        <th width="90">Mali GPU</th>
        <th width="90">Ethos U</th>
        <th width="90">Score (LER)</th>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/wav2letter/tflite_int8">Wav2letter INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.0877</td>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/wav2letter/tflite_pruned_int8">Wav2letter Pruned INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.0783</td>
    </tr>
</table>

**Dataset**: LibriSpeech

## Visual Wake Words

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="90">Cortex-A</th>
        <th width="90">Cortex-M</th>
        <th width="90">Mali GPU</th>
        <th width="90">Ethos U</th>
        <th width="90">Score (Accuracy)</th>
    </tr>
    <tr>
        <td><a href="models/visual_wake_words/micronet_vww2/tflite_int8">MicroNet VWW-2 INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.768</td>
    </tr>
    <tr>
        <td><a href="models/visual_wake_words/micronet_vww3/tflite_int8">MicroNet VWW-3 INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.855</td>
    </tr>
    <tr>
        <td><a href="models/visual_wake_words/micronet_vww4/tflite_int8">MicroNet VWW-4 INT8</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">:heavy_check_mark:</td>
        <td align="center">0.822</td>
    </tr>
</table>

**Dataset**: Visual Wake Words


### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.
* `*` - Code to recreate model available.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) unless otherwise explicitly stated.
