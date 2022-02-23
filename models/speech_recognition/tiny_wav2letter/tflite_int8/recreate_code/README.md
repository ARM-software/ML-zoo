# Tiny Wav2letter FP32/INT8/INT8_Pruned  Model Re-Creation
This folder contains a script that allows for the model to be re-created from scratch.
## Datasets
Tiny Wav2Letter was trianed on both LibriSpeech dataset hosted on OpenSLR and fluent-speech-corpus dataset hosted on Kaggle.
Please note that fluent-speech-corpus dataset hosted on [Kaggle](https://www.kaggle.com/tommyngx/fluent-speech-corpus) is a licensed dataset.
## Requirements
The script in this folder requires that the following must be installed:
- Python 3.6
- Create new dir: fluent_speech_commands_dataset
- (LICENSED DATASET!!) Download and extract fluent-speech-corpus from: https://www.kaggle.com/tommyngx/fluent-speech-corpus to fluent_speech_commands_dataset dir

## Running The Script
To run the script, run the following in a terminal: `./recreate_model.sh`
