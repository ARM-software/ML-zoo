# Model Zoo
![version](https://img.shields.io/badge/version-21.08-0091BD)
> A collection of machine learning models optimized for Arm IP.


## Anomaly Detection

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (AUC)</th>
    </tr>
    <tr>
        <td><a href="models/anomaly_detection/micronet_large/tflite_int8">MicroNet Large INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.968</td>
    </tr>
    <tr>
        <td><a href="models/anomaly_detection/micronet_medium/tflite_int8">MicroNet Medium INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.963</td>
    </tr>
    <tr>
        <td><a href="models/anomaly_detection/micronet_small/tflite_int8">MicroNet Small INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
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
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (Top 1 Accuracy)</th>
    </tr>
    <tr>
        <td><a href="models/image_classification/mobilenet_v2_1.0_224/tflite_int8">MobileNet v2 1.0 224 INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.697</td>
    </tr>
    <tr>
        <td><a href="models/image_classification/mobilenet_v2_1.0_224/tflite_uint8">MobileNet v2 1.0 224 UINT8 </a></td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
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
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (Accuracy)</th>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_large/model_package_tf/model_archive/TFLite/tflite_int8">CNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.923</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_medium/model_package_tf/model_archive/TFLite/tflite_int8">CNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.905</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_small/model_package_tf/model_archive/TFLite/tflite_int8">CNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.902</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_large/model_package_tf/model_archive/TFLite/tflite_int8">DNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.860</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_medium/model_package_tf/model_archive/TFLite/tflite_int8">DNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.839</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_small/model_package_tf/model_archive/TFLite/tflite_int8">DNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.821</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_clustered_fp32">DS-CNN Large Clustered FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.948</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_clustered_int8">DS-CNN Large Clustered INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.939</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_int8">DS-CNN Large INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.945</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_medium/model_package_tf/model_archive/TFLite/tflite_int8">DS-CNN Medium INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.939</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_small/model_package_tf/model_archive/TFLite/tflite_int8">DS-CNN Small INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.931</td>
    </tr>
        <tr>
        <td><a href="models/keyword_spotting/ds_cnn_small/model_package_tf/model_archive/TFLite/tflite_int16">DS-CNN Small INT16 *</a></td>
        <td align="center">INT16</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.934</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_large/model_package_tf/model_archive/TFLite/tflite_fp32">CNN Large FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.934</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_medium/model_package_tf/model_archive/TFLite/tflite_fp32">CNN Medium FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.918</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/cnn_small/model_package_tf/model_archive/TFLite/tflite_fp32">CNN Small FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.922</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_large/model_package_tf/model_archive/TFLite/tflite_fp32">DNN Large FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.867</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_medium/model_package_tf/model_archive/TFLite/tflite_fp32">DNN Medium FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.850</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/dnn_small/model_package_tf/model_archive/TFLite/tflite_fp32">DNN Small FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.836</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_fp32">DS-CNN Large FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.950</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_medium/model_package_tf/model_archive/TFLite/tflite_fp32">DS-CNN Medium FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.943</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/ds_cnn_small/model_package_tf/model_archive/TFLite/tflite_fp32">DS-CNN Small FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.939</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/micronet_large/tflite_int8">MicroNet Large INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.965</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/micronet_medium/tflite_int8">MicroNet Medium INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.958</td>
    </tr>
    <tr>
        <td><a href="models/keyword_spotting/micronet_small/tflite_int8">MicroNet Small INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.953</td>
    </tr>
</table>

**Dataset**: Google Speech Commands Test Set

## Noise Suppression

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (Average Pesq)</th>
    </tr>
    <tr>
        <td><a href="models/noise_suppression/RNNoise/tflite_int8">RNNoise INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">2.945</td>
    </tr>
</table>

**Dataset**: Noisy Speech Database For Training Speech Enhancement Algorithms And Tts Models

## Object Detection

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (mAP)</th>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_fp32">SSD MobileNet v1 FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.210</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_int8">SSD MobileNet v1 INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.234</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/ssd_mobilenet_v1/tflite_uint8">SSD MobileNet v1 UINT8 *</a></td>
        <td align="center">UINT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">0.180</td>
    </tr>
    <tr>
        <td><a href="models/object_detection/yolo_v3_tiny/tflite_fp32">YOLO v3 Tiny FP32 *</a></td>
        <td align="center">FP32</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
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
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (LER)</th>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/wav2letter/tflite_int8">Wav2letter INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.0877</td>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/wav2letter/tflite_pruned_int8">Wav2letter Pruned INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.0783</td>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/tiny_wav2letter/tflite_int8">Tiny Wav2letter INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.0348</td>
    </tr>
    <tr>
        <td><a href="models/speech_recognition/tiny_wav2letter/tflite_pruned_int8">Tiny Wav2letter Pruned INT8 *</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.0283</td>
    </tr>
</table>

**Dataset**: LibriSpeech, Fluent Speech

## Superresolution

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="125">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (PSNR)</th>
    </tr>
    <tr>
        <td><a href="models/superresolution/SESR/tflite_int8">SESR INT8 **</a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: HERO</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">35.00dB</td>
    </tr>
</table>

**Dataset**: DIV2K

## Visual Wake Words

<table>
    <tr>
        <th width="250">Network</th>
        <th width="100">Type</th>
        <th width="160">Framework</th>
        <th width="120">Cortex-A</th>
        <th width="120">Cortex-M</th>
        <th width="120">Mali GPU</th>
        <th width="120">Ethos U</th>
        <th width="90">Score (Accuracy)</th>
    </tr>
    <tr>
        <td><a href="models/visual_wake_words/micronet_vww2/tflite_int8">MicroNet VWW-2 INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.768</td>
    </tr>
    <tr>
        <td><a href="models/visual_wake_words/micronet_vww3/tflite_int8">MicroNet VWW-3 INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.855</td>
    </tr>
    <tr>
        <td><a href="models/visual_wake_words/micronet_vww4/tflite_int8">MicroNet VWW-4 INT8 </a></td>
        <td align="center">INT8</td>
        <td align="center">TensorFlow Lite</td>
        <td align="center">:heavy_multiplication_x: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">:heavy_check_mark: </td>
        <td align="center">0.822</td>
    </tr>
</table>

**Dataset**: Visual Wake Words


### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.
* `*` - Code to recreate model available.
* `**` - This model has a large memory footprint – it will not run on all platforms.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) unless otherwise explicitly stated.
