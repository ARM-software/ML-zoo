# Wav2letter Pruned INT8

## Description
Wav2letter is a convolutional speech recognition neural network. This implementation was created by Arm, pruned to 50% sparisty, fine-tuned and quantized using the TensorFlow Model Optimization Toolkit.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be downloaded by running the script `get_class_labels.sh`.

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | e389797705f5f8a7973c3280954dd5cdf54284a1 |
|  Size (Bytes)       | 23815520 |
|  Provenance         | https://github.com/ARM-software/ML-zoo/tree/master/models/speech_recognition/wav2letter/tflite_pruned_int8 |
|  Paper              | https://arxiv.org/abs/1609.03193 |

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

## Accuracy
Dataset: LibriSpeech

| Metric | Value |
|--------|-------|
| LER | 0.07981431 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |
| Sparsity | 50% |

## Network Inputs
<table>
    <tr>
        <th width="200">Input Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>input_2_int8</td>
        <td>(1, 296, 39)</td>
        <td>Speech converted to MFCCs and quantized to INT8</td> 
    </tr>
</table>

## Network Outputs
<table>
    <tr>
        <th width="200">Output Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>Identity_int8</td>
        <td>(1, 1, 148, 29)</td>
        <td>A tensor of (batch, time, class probabilities) that represents the probability of each class at each timestep. Should be passed to a decoder e.g. ctc_beam_search_decoder.</td> 
    </tr>
</table>
