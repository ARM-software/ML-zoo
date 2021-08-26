# MicroNet VWW-2 INT8

## Description
This is a fully quantized version (asymmetrical int8) of the MicroNet VWW-2 model developed by Arm, from the MicroNets paper. It is trained on the 'Visual Wake Words' dataset, more information can be found here: https://arxiv.org/pdf/1906.05721.pdf.

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Related Materials
### Class Labels
The class labels associated with this model can be created by running the script `get_class_labels.sh`.

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 5d887ca438c0a7feeed3c8c22dce99b55565c8ea |
|  Size (Bytes)       | 280384 |
|  Provenance         | https://arxiv.org/pdf/2010.11267.pdf |
|  Paper              | https://arxiv.org/pdf/2010.11267.pdf |

## Performance
| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_multiplication_x:         |
| Cortex-M |:heavy_check_mark:         |
| Mali GPU |:heavy_check_mark:         |
| Ethos U  |:heavy_check_mark:         |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Accuracy
Dataset: Visual Wake Words

| Metric | Value |
|--------|-------|
| Accuracy | 0.768 |

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
        <td>(1, 50, 50, 1)</td>
        <td>A 50x50 input image.</td> 
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
        <td>(1, 2)</td>
        <td>Per-class confidence across the two classes (0=no person present, 1=person present).</td> 
    </tr>
</table>
