# MicroNet Small INT8

## Description
This is a fully quantized version (asymmetrical int8) of the MicroNet Small model developed by Arm, from the MicroNets paper. It is trained on the 'slide rail' task from http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be created by running the script `get_class_labels.sh`.

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 6dc73515caea226065c3408d82d857b9908e3ffa |
|  Size (Bytes)       | 252848 |
|  Provenance         | https://arxiv.org/pdf/2010.11267.pdf |
|  Paper              | https://arxiv.org/pdf/2010.11267.pdf |

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_multiplication_x:         |
| Cortex-M |:heavy_check_mark:         |
| Mali GPU |:heavy_multiplication_x:         |
| Ethos U  |:heavy_check_mark:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Accuracy
Dataset: Dcase 2020 Task 2 Slide Rail

| Metric | Value |
|--------|-------|
| AUC | 0.9548 |

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
        <td>input</td>
        <td>(1, 32, 32, 1)</td>
        <td>Input is 64 steps of a Log Mel Spectrogram using 64 mels resized to 32x32.</td> 
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
        <td>Identity</td>
        <td>(1, 8)</td>
        <td>Raw logits corresponding to different machine IDs being anomalous</td> 
    </tr>
</table>
