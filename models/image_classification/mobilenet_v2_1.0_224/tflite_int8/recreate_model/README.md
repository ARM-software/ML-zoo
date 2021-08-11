# MobileNet v2 INT8 Re-Creation
This folder contains scripts that allow you to re-create the model and benchmark it's performance.

## Requirements
The scripts in this folder requires that the following must be installed:
- Python 3.7

## Required Datasets
Quantising and Benchmarking the model requires ImageNet Validation Set (ILSVRC2012). This can either be provided as a TFRecord file or in the form of images, with a text file providing the corresponding class labels. The script "write_tfrecord.py" then writes the images to the required TFRecord format for the quantisation scripts.
In the case of raw images, the images themselves should be stored in 'data --> validation_data --> validation_images --> ILSVRC_val_00000001.JPEG, ILSVRC_val_00000002.JPEG ...' while the text file should be saved as 'data --> validation_data --> val.txt'. "write_tfrecord.py" will create a TFRecord file "data --> validation_set --> validation-dataset.tfrecord"
In the case of the ImageNet Validation Set already being present as TFRecords, save them as 'data --> validation_data --> validation-dataset1.tfrecord, validation-dataset2.tfrecord ..."

## Running The Script
### Recreate The Model
Run the following command in a terminal: `./quantize_mobilenet_v2.sh`

### Benchmarking The Model
Run the following command in a terminal: `./benchmark_mobilenet_v2.sh`
