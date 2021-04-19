# Wav2letter INT8

## Description
Wav2letter is a convolutional speech recognition neural network. This implementation was created by Arm and quantized to the INT8 datatype.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

## Network Information
| Network Information |  Value         |
|---------------------|------------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 481b7621801363b64dca2cc02b661b26866af76c |
|  Size (Bytes)       | 23815520 |
|  Provenance         | https://github.com/ARM-software/ML-zoo/tree/master/models/speech_recognition/wav2letter/tflite_int8 |
|  Paper              | https://arxiv.org/abs/1609.03193 |

## Accuracy
Dataset: Librispeech

| Metric | Value |
|--------|-------|
| Ler | 0.08771 |

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:         |
| Cortex-M |:heavy_check_mark:         |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_check_mark:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.



## Optimizations
| Optimization |  Value  |
|-----------------|---------|
| Quantization | INT8 |

## Network Inputs
| Input Node Name |  Shape  | Description |
|-----------------|---------|-------------|
| input_2_int8 | (1, 296, 39) | Speech converted to MFCCs and quantized to INT8. |

## Network Outputs
| Output Node Name |  Shape  | Description |
|------------------|---------|-------------|
| Identity_int8 | (1, 1, 148, 29) | A tensor of time and class probabilities, that represents the probability of each class at each timestep. Should be passed to a decoder. For example ctc_beam_search_decoder. |
