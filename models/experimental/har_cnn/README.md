# har_int8.tflite

## Description
Model internally developed.
Based on dataset https://www.cis.fordham.edu/wisdm/dataset.php

## License
Apache v2

## Network Inputs
| Input Node Name | Shape | Example Path | Example Type | Example Use Case |
|-----------------|-------|--------------|------------------|--------------|
| conv2d_input | (1, 90, 3, 1) | models/ |  | Accelerometer data of someone walking. |

## Network Outputs
| Output Node Name | Shape | Description |
|------------------|-------|-------------|
| Identity | (1, 6) | Class probability of 6 classes |
