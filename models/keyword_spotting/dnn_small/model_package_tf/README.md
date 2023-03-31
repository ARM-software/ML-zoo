# DNN Small model package

This folder contains code that will allow you to recreate the DNN Small keyword spotting model from
the [Hello Edge paper](https://arxiv.org/pdf/1711.07128.pdf).

## License
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)

## Model Package Overview
| Model           	|   DNN_Small                            	   |
|:---------------:	|:------------------------------------------:|
| <u>**Format**</u>:          	| Keras, Saved Model, TensorFlow Lite int8, TensorFlow Lite fp32 |
| <u>**Feature**</u>:         	|   Keyword spotting for Arm Cortex-M CPUs   |
| <u>**Architectural Delta w.r.t. Vanilla**</u>: |                    None                    |
| <u>**Domain**</u>:         	|              Keyword spotting              |
| <u>**Package Quality**</u>: 	|                 Optimised                  |

## Model Recreation

In order to recreate the model you will first need to be using ```Python3.7``` and install the requirements in ```requirements.txt```.

Once you have these requirements satisfied you can execute the recreation script contained within this folder, just run:

```bash
bash ./recreate_model.sh
```

Running this script will use the pre-trained checkpoint files supplied in the ```./model_archive/model_source/weights``` folder
to generate the TFLite files and perform evaluation on the test sets. Both an fp32 version and a quantized version will be produced.
The quantized version will use post-training quantization to fully quantize it.

If you want to run training from scratch you can do this by supplying ```--train``` when running the script. For example:

```bash
bash ./recreate_model.sh --train
```

Training is then performed and should produce a model to the stated accuracy in this repository.
Note that exporting to TFLite will still happen with the pre-trained checkpoint files so you will need to re-run the script
and this time supply the path to the new checkpoint files you want to use, for example:

```bash
bash ./recreate_model.sh --ckpt <checkpoint_path>
```


## Training

To train a DNN with 3 fully-connected layers with 128 neurons in each layer, run:

```
python train.py --model_architecture dnn --model_size_info 128 128 128
```
The command line argument *--model_size_info* is used to pass the neural network layer
dimensions such as number of layers, convolution filter size/stride as a list to models.py,
which builds the TensorFlow graph based on the provided model architecture
and layer dimensions. For more info on *model_size_info* for each network architecture see
[models.py](models.py).

The training commands with all the hyperparameters to reproduce the models shown in the
[paper](https://arxiv.org/pdf/1711.07128.pdf) are given [here](recreate_model.sh).

## Testing
To run inference on the trained model from a checkpoint and get accuracy on validation and test sets, run:
```
python evaluation.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint <checkpoint_path>
```
The parameters used here should match those used in the Training step.

## Optimization

We introduce a new *optional* step to optimize the trained keyword spotting model for deployment.

Here we use TensorFlow's [weight clustering API](https://www.tensorflow.org/model_optimization/guide/clustering) to reduce the compressed model size and optimize inference on supported hardware. 32 weight clusters and kmeans++ cluster intialization method are used as the clustering hyperparameters.

To optimize your trained model (e.g. a DNN), a trained model checkpoint is needed to run clustering and fine-tuning on.
You can use the pre-trained checkpoints provided, or train your own model and use the resulting checkpoint.

To apply the optimization and fine-tuning, run the following command:
```
python optimisations.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint <checkpoint_path>
```
The parameters used here should match those used in the Training step, except for the number of training steps.
The number of training steps is reduced since the optimization step only requires fine-tuning.

This will generate a clustered model checkpoint that can be used in the quantization step to generate a quantized and clustered TFLite model.

## Quantization and TFLite Conversion

As part of the update we now use TensorFlow's
[post training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) to
make quantization of the trained models super simple.

To quantize your trained model (e.g. a DNN) run:
```
python convert_to_tflite.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint <checkpoint_path> [--inference_type int8|int16]
```
The parameters used here should match those used in the Training step.

The inference_type parameter is *optional* and to be used if a fully quantized model with inputs and outputs of type int8 or int16 is needed. It defaults to fp32.

This step will produce a quantized TFLite file *dnn_quantized.tflite*.
You can test the accuracy of this quantized model on the test set by running:
```
python evaluation.py --tflite_path dnn_quantized.tflite
```
The parameters used here should match those used in the Training step.

`convert_to_tflite.py` uses post-training quantization to generate a quantized model by default. If you wish to convert to a floating point TFLite model, use the command below:

```
python convert_to_tflite.py --model_architecture dnn --model_size_info 128 128 128 --checkpoint <checkpoint_path> --no-quantize
```

This will produce a floating point TFLite file *dnn.tflite*. You can test the accuracy of this floating point model using `evaluation.py` as above.
