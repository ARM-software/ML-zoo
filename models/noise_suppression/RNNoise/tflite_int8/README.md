# RNNoise INT8

## Description
RNNoise is a noise reduction network, that helps to remove noise from audio signals while maintaining any speech. This is a TFLite quantized version that takes traditional signal processing features and outputs gain values that can be used to remove noise from audio. It also detects if voice activity is present.
This is a 1 step model trained on Noisy speech database for training speech enhancement algorithms and TTS models that requires hidden states to be fed in at each time step.
Dataset license link: https://datashare.ed.ac.uk/handle/10283/2791
This model is converted from FP32 to INT8 using post-training quantization.


## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 2d973fe7116e0bc3674f0f3f0f7185ffe105bba5 |
|  Size (Bytes)       | 113472 |
|  Provenance         | https://arxiv.org/pdf/1709.08243.pdf |
|  Paper              | https://arxiv.org/pdf/1709.08243.pdf |

## Performance

| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:          |
| Cortex-M |:heavy_check_mark:          |
| Mali GPU |:heavy_check_mark:          |
| Ethos U  |:heavy_check_mark:          |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Accuracy
Dataset: Noisy Speech Database For Training Speech Enhancement Algorithms And Tts Models

| Metric | Value |
|--------|-------|
| Average Pesq | 2.945 |

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
        <td>main_input_int8</td>
        <td>(1, 1, 42)</td>
        <td>Pre-processed signal features extracted from 480 values of a 48KHz wav file</td> 
    </tr>
    <tr>
        <td>vad_gru_prev_state_int8</td>
        <td>(1, 24)</td>
        <td>Previous GRU state for the voice activity detection GRU</td> 
    </tr>
    <tr>
        <td>noise_gru_prev_state_int8</td>
        <td>(1, 48)</td>
        <td>Previous GRU state for the noise GRU</td> 
    </tr>
    <tr>
        <td>denoise_gru_prev_state_int8</td>
        <td>(1, 96)</td>
        <td>Previous GRU state for the denoise GRU</td> 
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
        <td>(1, 1, 96)</td>
        <td>Next GRU state for the denoise GRU</td> 
    </tr>
    <tr>
        <td>Identity_1_int8</td>
        <td>(1, 1, 22)</td>
        <td>Gain values that can be used to remove noise from this audio sample</td> 
    </tr>
    <tr>
        <td>Identity_2_int8</td>
        <td>(1, 1, 48)</td>
        <td>Next GRU state for the noise GRU</td> 
    </tr>
    <tr>
        <td>Identity_3_int8</td>
        <td>(1, 1, 24)</td>
        <td>Next GRU state for the voice activity detection GRU</td> 
    </tr>
    <tr>
        <td>Identity_4_int8</td>
        <td>(1, 1, 1)</td>
        <td>Probability that this audio sample contains voice activity</td> 
    </tr>
</table>
