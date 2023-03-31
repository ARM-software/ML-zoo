# keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_fp32

## Description
This is a floating point fp32 version of the DS-CNN Large model developed by Arm, from the Hello Edge paper.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  Datatype           | fp32 |
|  SHA-1 Hash         | fea0e0dc13fc4207dd44904fe701f34254dd4767 |
|  Size (Bytes)       | 1652648 |
|  Provenance         | https://arxiv.org/abs/1711.07128 |
|  Training           | Trained by Arm |
|  Paper | https://arxiv.org/abs/1711.07128 |

## DataSet
| Dataset Information | Value |
|--------|-------|
| Name | Google Speech Commands test set |

## Accuracy

| Metric | Value |
|--------|-------|
| accuracy | 95.03% |

## HW Support
| HW Support   | Value |
|--------------|-------|
| Cortex-A |:heavy_check_mark:         |
| Cortex-M |:heavy_check_mark: HERO        |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_multiplication_x:          |

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
|  Quantization - full    | :heavy_multiplication_x:    |

## Network Inputs
| Input Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| input | (1, 490) | int8 | models/keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_fp32/testing_input/input | fp32 | [1, 490] | The input is a processed MFCCs |

## Network Outputs
| Output Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| Identity | (1, 12) | int8 | models/keyword_spotting/ds_cnn_large/model_package_tf/model_archive/TFLite/tflite_fp32/testing_output/Identity | fp32 | [1, 12] | The probability on 12 keywords |