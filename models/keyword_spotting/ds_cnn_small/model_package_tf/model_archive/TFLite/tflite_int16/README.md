# keyword_spotting/ds_cnn_small/model_package_tf/model_archive/TFLite/tflite_int16

## Description
This is a fully quantized int16 version of the DS-CNN Small model developed by Arm, from the Hello Edge paper.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  Datatype           | int16 |
|  SHA-1 Hash         | e82c7d645bec3dec580a096de0a297c6dd9a6463 |
|  Size (Bytes)       | 55392 |
|  Provenance         | https://github.com/ARM-software/ML-examples/tree/main/tflu-kws-cortex-m |
|  Training           | Trained by Arm |
|  Paper | https://arxiv.org/abs/1711.07128 |

## DataSet
| Dataset Information | Value |
|--------|-------|
| Name | Google Speech Commands test set |

## Accuracy

| Metric | Value |
|--------|-------|
| Accuracy | 93.39% |

## HW Support
| HW Support   | Value |
|--------------|-------|
| Cortex-A |:heavy_check_mark:          |
| Cortex-M |:heavy_check_mark: HERO         |
| Mali GPU |:heavy_check_mark:          |
| Ethos U  |:heavy_check_mark:          |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Network Quality
| Network Quality         | Value |
|-------------------------|-------|
|  Recreate               | :heavy_check_mark:    |
|  Quality level          | Hero    |
|  Vanilla                | :heavy_check_mark:    |
|  Clustered              | :heavy_multiplication_x:    |
|  Pruned                 | :heavy_multiplication_x:    |
|  Quantization - default | :heavy_multiplication_x:    |
|  Quantization - full    | :heavy_check_mark:    |

## Network Inputs
| Input Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| serving_default_input:0 | (1, 490) | int16 | models/keyword_spotting/ds_cnn_small/model_package_tf/model_archive/TFLite/tflite_int16/testing_input | int16 | [1, 490] | The input is a processed MFCCs |

## Network Outputs
| Output Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| StatefulPartitionedCall:0 | (1, 12) | int16 | models/keyword_spotting/ds_cnn_small/model_package_tf/model_archive/TFLite/tflite_int16/testing_output | int16 | [1, 12] | The probability on 12 keywords |