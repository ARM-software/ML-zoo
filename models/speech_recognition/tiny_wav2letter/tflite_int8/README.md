# Tiny Wav2letter INT8

## Description
Tiny Wav2letter is a tiny version of the original Wav2Letter model. It is a convolutional speech recognition neural network. This implementation was created by Arm, pruned to 50% sparsity, fine-tuned and quantized using the TensorFlow Model Optimization Toolkit.



## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 13ca2294ba4bbb1f1c6c5e663cb532d58cd76a6b |
|  Size (Bytes)       | 3997112 |
|  Provenance         | https://github.com/ARM-software/ML-zoo/tree/master/models/speech_recognition/wav2letter |
|  Paper              | https://arxiv.org/abs/1609.03193 |

## Performance

| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:          |
| Cortex-M |:heavy_check_mark:          |
| Mali GPU |:heavy_multiplication_x:          |
| Ethos U  |:heavy_check_mark:          |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Accuracy
Dataset: Fluent Speech (trianed on LibriSpeech,Mini LibrySpeech,Fluent Speech)
<br />
Please note that Fluent Speech dataset hosted on Kaggle is a licensed dataset.

| Metric | Value |
|--------|-------|
| LER | 0.0348 |
| WER | 0.112 |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |

## Network Inputs
<table>
    <tr>
        <th width="200">Input Node Name</th>
        <th width="100">Shape</th>
        <th width="300">Description</th>
    </tr>
    <tr>
        <td>input_1_int8</td>
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
        <td>A tensor of time and class probabilities, that represents the probability of each class at each timestep. Should be passed to a decoder. For example ctc_beam_search_decoder.</td> 
    </tr>
</table>
