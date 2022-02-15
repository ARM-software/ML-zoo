# Model Development Guide

This document describes the process of training a model from scratch, using the Tiny Wav2Letter model as an example.

## Datasets

The first thing to decide is which dataset the model is to be trained on. Most commonly used datasets can either be found online or in the ARM AWS S3 bucket. In the case of Tiny Wav2Letter, both the LibriSpeech dataset hosted on [OpenSLR](http://www.openslr.org/resources.php) and fluent-speech-corpus dataset hosted on [Kaggle](https://www.kaggle.com/tommyngx/fluent-speech-corpus) were used to train the model.
! ! please note that fluent-speech-corpus dataset hosted on [Kaggle](https://www.kaggle.com/tommyngx/fluent-speech-corpus) is a licensed dataset.

## Preprocessing

The dataset is often not in the right format for training, so preprocessing steps must be taken. In this case, the LibriSpeech dataset consists of audio files, however the paper stated that as input MFCCs should be used, so the audio files needed to be converted. It is to be recommended that all preprocessing be performed offline, as this will make the actual training process faster, as the data is already in the correct format. The most convenient way to store the preprocessed data is using [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord), as these are very easily loaded into TFDatasets. While it can take a long time to write the whole dataset to a TFRecord file, it is outweighed by the time saved during training.
Please note: input audio data sample rate is 16K
## Model Architecture

The model architecture can generally be found from a variety of sources. If a similar model exists in the IMZ, then [Netron](https://netron.app) can be used to inspect the TFLite file. The original paper the model was proposed in will also define the architecture. The model should ideally be defined using the Tensorflow Functional API rather than the sequential API.

## Loss Function and Metrics

The loss function and desired metrics will be defined by the model. If at all possible, structure the data such that the input to the loss function is in the form (y_true, y_predicted) as this will enable model.fit to be used and avoid custom training loops. TensorFlow has lots of standard loss functions straight out of the box, but if need be custom loss functions can be defined, as was the case in TinyWav2Letter.

## Model Training

If everything else has been set up properly, the code here should not be complicated. Load the datasets, create an instance of the model, and then ideally run model.fit but if that's not possible use tf.GradientTape. Use callbacks to write to a log directory (tf.keras.callbacks.Tensorboard), then use Tensorboard to visualise the training process. One can use the [TensorFlow Profiler](https://www.tensorflow.org/guide/profiler) to identify bottlenecks in the training pipeline and speed up the training process. Another useful callback is tf.keras.callbacks.ModelCheckpoint which saves a checkpoint at defined intervals, so one can pick up training from where it was left off. Generally we will want a training set, a validation set and a test set, with normally about a 90:5:5 split. If the model performs well on the training set but not on the validation set or test set, then the model is overfitting. This can be reduced by introducing regularisation, increasing the amount of data, reducing model complexity or adjusting hyperparameters. In the case of TinyWav2Letter, the model was initially trained on the full-size LibriSpeech Dataset to capture the features of speech, then fine-tuned on the much smaller Mini LibriSpeech to improve the accuracy on the smaller dataset, then fine-tuned on fluent-speech-corpus dataset

## Optimisation and Conversion

Once the model has been trained to satisfaction, it can be optionally be optimised using the TensorFlow Model Optimization Toolkit. Pruning sets a specified percentage of the weights  to be 0, so the model is sparser, which can lead to faster inference. Clustering clusters together weights of similar values, reducing the number of unique values. This again leads to faster inference. Quantisation (eg to INT8) converts all the weights to INT8 representations, giving a 4x reduction in size compared to FP32. If the quantisation process affects the metric too severely, quantisation aware training can be performed, which fine-tunes the model and makes it more robust to quantisation. Quantisation aware training requires at least Tensorflow 2.5.0. The final step is to convert the model to the TFLite model format. If using INT8 conversion, one must define a representative dataset to calibrate the model.

## Training a smaller FP32 Keras Model

The trained Wav2Letter model then serves as the foundation for investigations into how best to reduce the size of the network.  

There are three hyperparameters relevant to the size of the network. They are:

- Number of layers
- Number of filters in each convolutional layer
- Stride of each convolutional filter

The following table shows chose architecture for Tiny Wav2Letter and the effect that it has on the output size. 

| Identifier | Total Number of Layers | Number of middle Convolutional Layers | Corresponding number of filters  | Number of filters in the antepenultimate Conv2D Layer | Number of filters in the penultimate Conv2D Layer |
| ------ | ------ | ------ | ------ | ------ | ------ |
| Wav2Letter | 11 | 7 | 250 | 2000 | 2000 |
| Tiny Wav2Letter | 6 | 5 | 100 | 750 | 750 |

| Identifier | Size (MB) | LER | WER |
| ------ | ------ | ------ | ------ |
| Wav2Letter INT8| 22.7 | 0.0877** | N/A |
| Wav2Letter INT8 pruned|  22.7 | 0.0783** | N/A |
| Tiny Wav2Letter FP32| 15.6* | 0.0351 | 0.0714 |
| Tiny Wav2Letter FP32 pruned| 15.6* | 0.0266 | 0.0577 |
| Tiny Wav2Letter INT8| 3.81 | 0.0348 | 0.1123 |
| Tiny Wav2Letter INT8 pruned| 3.81 | 0.0283 | 0.0886 |

"*" - the size is according to the tflite model \
**  trained on different dataset

