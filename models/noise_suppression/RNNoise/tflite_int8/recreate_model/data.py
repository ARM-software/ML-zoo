#  Copyright (c) 2021 Arm Limited. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Code related to preparing raw training data and producing tf Datasets for training RNNoise."""
from pathlib import Path
import logging
import argparse

import librosa
import numpy as np
import h5py
import tensorflow as tf
import soundfile as sf

from rnnoise_pre_processing import RNNoisePreProcess


def get_tf_dataset_from_h5(h5_path, window_size, batch_size):
    """Returns tf Datasets for training RNNoise.

    This function expects h5 files produced by generate_h5_files() or from original RNNoise repository.
    """
    with h5py.File(h5_path, 'r') as hf:
        all_data = hf['data'][:]

    nb_sequences = len(all_data) // window_size
    logging.info(f"Number of training sequences: {nb_sequences}")

    # We need data in the order (batch, sequence, features)
    x = all_data[:nb_sequences*window_size, :42]
    x = np.reshape(x, (nb_sequences, window_size, 42))

    y = all_data[:nb_sequences*window_size, 42:64]
    y = np.reshape(y, (nb_sequences, window_size, 22))

    vad_y = all_data[:nb_sequences*window_size, 86:87]
    vad_y = np.reshape(vad_y, (nb_sequences, window_size, 1))

    with tf.device('/CPU:0'):
        tf_dataset = tf.data.Dataset.from_tensor_slices((x, (y, vad_y)))
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return tf_dataset


def load_wav(wav_path):
    """Load audio file and transform to int16 range but as fp32 type."""
    loaded_audio = librosa.load(wav_path, sr=48000)[0]
    loaded_audio = loaded_audio * (2 ** 15)

    return loaded_audio


def load_clean_noisy_wavs(clean_speech_folder, noisy_speech_folder, isolate_noise):
    """Return lists of loaded audio for folders of clean and noisy audio.

    It is expected the clean speech folder and noisy speech folder have wav files with the same names.
    The wav files in the noisy speech folder will have the same speech but overlaid with some noise.

    If isolate_noise is True then this function will also isolate and return the noise along with the
    clean and noisy speech."""
    clean_speech_audio = []
    noise_speech_audio = []
    isolated_noise_audio = []

    for clean_file in clean_speech_folder.rglob('*.wav'):
        try:
            clean_wav = librosa.load(clean_file, sr=48000)[0]
            noisy_file_path = noisy_speech_folder / clean_file.name
            noisy_wav = librosa.load(noisy_file_path, sr=48000)[0]

            if isolate_noise:
                noise_sample = noisy_wav - clean_wav
                noise_sample *= 2**15
                isolated_noise_audio.append(noise_sample)

            # Convert audio from fp32 range into int16 range (but as fp32 type).
            clean_wav *= 2**15
            clean_speech_audio.append(clean_wav)
            noisy_wav *= 2**15
            noise_speech_audio.append(noisy_wav)
        except:
            logging.warning(f"Could not process {clean_file}, make sure it exists in both clean and noisy folders.")
            pass

    return clean_speech_audio, noise_speech_audio, isolated_noise_audio


def create_combined_wavs(clean_speech_folder, noisy_speech_folder):
    """Load folders of clean and noisy speech then combine them into two wav files of clean speech and isolated noise.

    It is expected the clean speech folder and noisy speech folder have wav files with the same names.
    The wav files in the noisy speech folder will have the same speech but overlaid with some noise."""
    clean_speech_audio_list, _, noise_audio_list = load_clean_noisy_wavs(Path(clean_speech_folder),
                                                                         Path(noisy_speech_folder),
                                                                         isolate_noise=True)
    clean_speech_audio = np.concatenate(clean_speech_audio_list, axis=0)
    noise_audio = np.concatenate(noise_audio_list, axis=0)

    clean_speech_audio = np.rint(clean_speech_audio).astype(np.int16)
    noise_audio = np.rint(noise_audio).astype(np.int16)

    sf.write("combined_speech.wav", clean_speech_audio, 48000, 'PCM_16')
    sf.write("combined_noise.wav", noise_audio, 48000, 'PCM_16')


def generate_h5_files():
    """Generate the training and testing h5 files for RNNoise.

    This function is too slow to generate a full training set, it is recommended to use the original RNNoise
    repository to generate your training h5 files."""
    max_count_train = 50400000  # Corresponds to 140 hours of audio.
    max_count_test = 500000

    # Training h5 file creation.
    clean_speech_audio_list, _, noise_audio_list = load_clean_noisy_wavs(Path(FLAGS.clean_train_wav_folder),
                                                                         Path(FLAGS.noisy_train_wav_folder),
                                                                         isolate_noise=True)
    clean_speech_audio = np.concatenate(clean_speech_audio_list, axis=0)
    noise_audio = np.concatenate(noise_audio_list, axis=0)

    preprocess = RNNoisePreProcess(training=True)
    input_features, output_labels, vad_labels = preprocess.get_training_features(clean_speech_audio,
                                                                                 noise_audio,
                                                                                 max_count_train)

    # Re-use output labels as dummy replacement for Log energy labels.
    # Log energy labels are not used in training, this is only done so our h5 looks similar to RNNoise original h5.
    dummy_Ln_labels = output_labels
    h5f = h5py.File('train.h5', 'w')
    h5f.create_dataset('data', data=np.concatenate(input_features, output_labels, dummy_Ln_labels, vad_labels, axis=1))
    h5f.close()

    # Testing h5 file creation.
    clean_speech_audio_list, _, noise_audio_list = load_clean_noisy_wavs(Path(FLAGS.clean_test_wav_folder),
                                                                         Path(FLAGS.noisy_test_wav_folder),
                                                                         isolate_noise=True)
    clean_speech_audio = np.concatenate(clean_speech_audio_list, axis=0)
    noise_audio = np.concatenate(noise_audio_list, axis=0)

    preprocess = RNNoisePreProcess(training=True)
    input_features, output_labels, vad_labels = preprocess.get_training_features(clean_speech_audio,
                                                                                 noise_audio,
                                                                                 max_count_test)
    dummy_Ln_labels = output_labels
    h5f = h5py.File('test.h5', 'w')
    h5f.create_dataset('data', data=np.concatenate(input_features, output_labels, dummy_Ln_labels, vad_labels, axis=1))
    h5f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clean_train_wav_folder',
        type=str,
        required=True,
        help='Path to the training set of clean wav files.'
    )
    parser.add_argument(
        '--noisy_train_wav_folder',
        type=str,
        required=True,
        help='Path to the training set of noisy wav files.'
    )
    parser.add_argument(
        '--clean_test_wav_folder',
        type=str,
        required=True,
        help='Path to the testing set of clean wav files.'
    )
    parser.add_argument(
        '--noisy_test_wav_folder',
        type=str,
        required=True,
        help='Path to the testing set of noisy wav files.'
    )
    FLAGS, _ = parser.parse_known_args()
    generate_h5_files()
