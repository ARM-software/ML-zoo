# keyword_spotting/cnn_medium/model_package_tf/model_archive/TFLite/tflite_int8

## Description
This is a fully quantized int8 version of the CNN Medium model developed by Arm, from the Hello Edge paper.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information | Value |
|---------------------|-------|
|  Framework          | TensorFlow Lite |
|  Datatype           | int8 |
|  SHA-1 Hash         | 6bc68074d960bbb0c695e19fd96fd7903131ef60 |
|  Size (Bytes)       | 186064 |
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
| Accuracy | 90.47% |

## HW Support
| HW Support   | Value |
|--------------|-------|
| Cortex-A |:heavy_check_mark:          |
| Cortex-M |:heavy_check_mark:          |
| Mali GPU |:heavy_check_mark:          |
| Ethos U  |:heavy_check_mark:          |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Network Quality
| Network Quality         | Value |
|-------------------------|-------|
|  Recreate               | :heavy_check_mark:    |
|  Quality level          | Deployable    |
|  Vanilla                | :heavy_check_mark:    |
|  Clustered              | :heavy_multiplication_x:    |
|  Pruned                 | :heavy_multiplication_x:    |
|  Quantization - default | :heavy_multiplication_x:    |
|  Quantization - full    | :heavy_check_mark:    |

## Network Inputs
| Input Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| input | (1, 490) | int8 | models/keyword_spotting/cnn_medium/model_package_tf/model_archive/TFLite/tflite_int8/testing_input/input | fp32 | [1, 490] | The input is a processed MFCCs |

## Network Outputs
| Output Node Name | Shape | Type | Example Path | Example Type | Example Shape | Example Use Case |
|-----------------|-------|-------|--------------|-------|-------|-----------------|
| Identity | (1, 12) | int8 | models/keyword_spotting/cnn_medium/model_package_tf/model_archive/TFLite/tflite_int8/testing_output/Identity | fp32 | [1, 12] | The probability on 12 keywords |