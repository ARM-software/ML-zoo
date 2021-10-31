# Recreate RNNoise model

This folder contains code that will allow you to recreate the model and benchmark the performance.

## How to train

In order to recreate the RNNoise model exactly this will require you download the same dataset that we used to train
the model. Downloading and unzipping the data can be done by running the following command: 

```bash
./get_data.sh
```

This will download the [Noisy speech database for training speech enhancement algorithms and TTS models](
https://datashare.ed.ac.uk/handle/10283/2791) and unzip it into training and testing folders for both clean and
noisy data.

Next you will need to create test and training .h5 files. These will contain the input features and labels for
training the model.

You have two methods of doing this:

The first, and recommended way, is to follow the first 3 instructions found in the original RNNoise repository [here](
https://github.com/xiph/rnnoise/blob/master/TRAINING-README) to generate h5 files. You will need to combine all
the clean audio in to one wav file and all the isolated noise audio into another. We provide an example function that
can do this, see: ```create_combined_wavs``` in ```data.py```: 

Alternatively, you can use our Python implementation like so:
```bash
python data.py --clean_train_wav_folder=./clean_trainset_56spk_wav --noisy_train_wav_folder=./noisy_trainset_56spk_wav 
--clean_test_wav_folder=./clean_testset_wav --noisy_test_wav_folder=./noisy_testset_wav
```

However, this is much much slower than the original implementation.

After you have train and test h5 files created you can run the following shell script to perform training and generate
the final TFLite files.

```bash
./train_and_quantise_model.sh
```

This shell script expects that your training h5 file is called ```train.h5``` and your testing h5 file is
called ```test.h5```

Finally, to evaluate performance of the models you can run the following Python script:
```bash
python test.py --clean_wav_folder=./clean_testset_wav --noisy_wav_folder=./noisy_testset_wav --tflite_path=<path_to_tflite>
```

This evaluation may take some time to complete.