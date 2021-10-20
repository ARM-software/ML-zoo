# SESR INT8

## Description
SESR, super-efficient super resolution (formerly known as FSRCNNv5) is a network aims to generate a high-resolution image from a low-resolution input.
Name was changed by ARM developers when they wrote research paper on their technique.
The attached int8 fully quantized tflite model achieves 35.00dB PSNR on DIV2K dataset. 
The model takes 1080p input (in YCbCr, i.e., takes a 1x1920x1080x1 tensor as input) and outputs 4K images (in YCbCr, i.e., 1x3840x2160x1 output).  
Compatability note:  
Please note that SESR is a high-end network operating on 1080p->4K images and runtime memory use of this network requires an end system with at least 100MB of memory available to ensure successful execution.
We anticipate the network being used in premium devices as part of a camera imaging pipeline providing highest quality digital zoom.

 Submitter:
 chen.hayat@arm.com

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Network Information
| Network Information |  Value         |
|---------------------|----------------|
|  Framework          | TensorFlow Lite |
|  SHA-1 Hash         | 5abc5f05202aa1b0b9c34c5a978b6aa0a02f7ec5 |
|  Size (Bytes)       | 23680 |
|  Provenance         | https://arxiv.org/abs/2103.09404 |
|  Paper              | https://arxiv.org/abs/2103.09404 |

## Performance
This model has a large memory footprint – it will not run on all platforms.

| Platform | Optimized |
|----------|:---------:|
| Cortex-A |:heavy_check_mark:          |
| Cortex-M |:heavy_multiplication_x:          |
| Mali GPU |:heavy_check_mark:          |
| Ethos U  |:heavy_multiplication_x:          |

### Key
* :heavy_check_mark: - Will run on this platform.
* :heavy_multiplication_x: - Will not run on this platform.

## Accuracy
Dataset: DIV2K

| Metric | Value |
|--------|-------|
| PSNR | 35.00dB |

## Optimizations
| Optimization |  Value  |
|--------------|---------|
| Quantization | INT8 |

## Network Inputs
<table>
    <tr>
        <th width="200">Input Node Name</th>
        <th width="100">Shape</th>
        <th width="400">Description</th>
    </tr>
    <tr>
        <td>net_input</td>
        <td>(1, 1920, 1080, 1)</td>
        <td>Low-resolution input: 1080p (in YCbCr, i.e., take a 1x1920x1080x1 tensor as input) </td> 
    </tr>
</table>

## Network Outputs
<table>
    <tr>
        <th width="200">Output Node Name</th>
        <th width="100">Shape</th>
        <th width="400">Description</th>
    </tr>
    <tr>
        <td>net_output</td>
        <td>(1, 3840, 2160, 1)</td>
        <td>High-resolution input: 4K images (in YCbCr, i.e., 1x3840x2160x1 output).</td> 
    </tr>
</table>
