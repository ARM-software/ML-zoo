# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas
import os
import librosa
import tensorflow as tf


def normalize(values):
    """ Normalize values to mean 0 and std 1. """
    return (values - np.mean(values)) / np.std(values)


def overlap(batch_x, n_context=296, n_input=39):
    """
    Due to the requirement of static shapes(see fix_batch_size()),
    we need to stack the dynamic data to form a static input shape.
    Using the n_context of 296 (1 second of mfcc)
    """
    window_width = n_context
    num_channels = n_input

    batch_x = tf.expand_dims(batch_x, axis=0)
    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(
        np.eye(window_width * num_channels).reshape(
            window_width, num_channels, window_width * num_channels
        ),
        tf.float32,
    )
    # Create overlapping windows
    batch_x = tf.nn.conv1d(batch_x, eye_filter, stride=1, padding="SAME")
    # Remove dummy depth dimension and reshape into
    # [n_windows, n_input]
    batch_x = tf.reshape(batch_x, [-1, num_channels])

    return batch_x


def label_from_string(str_to_label: dict, string: str) -> int:
    try:
        return str_to_label[string]
    except KeyError as e:
        raise KeyError(
            f"ERROR: String: {string} in transcripts not occur in alphabet!"
        ).with_traceback(e.__traceback__)


def text_to_int_array_wrapper(alphabet_dict: dict):
    def text_to_int_array(original: str):
        r"""
        Given a Python string ``original``, map characters
        to integers and return a numpy array representing the processed string.
        """

        return np.asarray([label_from_string(alphabet_dict, c) for c in original])
    return text_to_int_array


class LibriSpeechMfcc:
    def __init__(self, data_dir: str):
        """
            Args:
                data_dir: Absolute path to librispeech data folder
        """
        self.overlap = False
        self.data_dir = data_dir
        self.seed = 0
        self.train = False
        self.batch_size = 32
        self.num_samples = 0
        self.input_files = []

    def training_set(self, overlap=False, batch_size=32):
        """
        Args:
            overlap: boolean to create overlapping windows
            batch_size: batch size required for the set
        """
        self.input_files = [
            "librivox-train-clean-100.csv",
            "librivox-train-clean-360.csv",
            "librivox-train-other-500.csv",
        ]

        self.train = True
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_samples = 281241
        return self.create_dataset()

    def evaluation_set(self, overlap=False, batch_size=32):
        """
        Args:
            overlap: boolean to create overlapping windows
            batch_size: batch size required for the set
        """
        self.input_files = ["librivox-test-clean.csv"]

        self.train = False
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_samples = 2620
        return self.create_dataset()

    def validation_set(self, overlap=False,  batch_size=32):
        """
        Args:
            overlap: boolean to create overlapping windows
            batch_size: batch size required for the set
        """
        self.input_files = ["librivox-dev-clean.csv"]

        self.train = False
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_samples = 2703
        return self.create_dataset()

    def create_dataset(self) -> tf.data.Dataset:
        """Create dataset generator for use in fit and evaluation functions."""
        df = self.read_input_files()
        df.sort_values(by="wav_filesize", inplace=True)

        # Convert to character index arrays
        alphabet = "abcdefghijklmnopqrstuvwxyz' @"
        alphabet_dict = {c: ind for (ind, c) in enumerate(alphabet)}
        df["transcript"] = df["transcript"].apply(text_to_int_array_wrapper(alphabet_dict))

        def generate_values():
            for _, row in df.iterrows():
                yield row.wav_filename, row.transcript

        dataset = tf.data.Dataset.from_generator(
            generate_values, output_types=(tf.string, tf.int32)
        )
        # librosa.feature.mfcc takes a long time to run when shuffling
        # so lets shuffle the data before performing our mapping function
        if self.train:
            dataset = dataset.shuffle(
                buffer_size=max(self.batch_size * 2, 1024), seed=self.seed
            )

        dataset = dataset.map(
            lambda filename, transcript: tf.py_function(
                self.load_data_mfcc,
                inp=[filename, transcript],
                Tout=[tf.float32, tf.int32],
            )
        )
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(tf.TensorShape([None, 39]), tf.TensorShape([None])),
            padding_values=(0.0, 28), drop_remainder=True
        )
        # Indication that shuffling is executed before mapping function
        return dataset

    def read_input_files(self) -> pandas.DataFrame:
        """Read the input files required for a particular set."""
        source_data = None
        for csv in self.input_files:
            file = pandas.read_csv(os.path.join(self.data_dir, csv), encoding="utf-8", na_filter=False)
            csv_dir = os.path.dirname(os.path.abspath(csv))
            file["wav_filename"] = file["wav_filename"].str.replace(
                r"(^[^/])", lambda m: os.path.join(csv_dir, m.group(1))
            )  # pylint: disable=cell-var-from-loop
            if source_data is None:
                source_data = file
            else:
                source_data = source_data.append(file)
        return source_data

    def num_steps(self, batch):
        """
        Get the number of steps based on the given batch size and the number
        of samples.
        """
        return int(np.math.ceil(self.num_samples / batch))

    def load_data_mfcc(self, filename, transcript):
        """ Calculate mfcc from the given raw audio data for Wav2Letter. """
        audio_data, samplerate = librosa.load(filename.numpy(), sr=16000)
        mfcc = librosa.feature.mfcc(
            audio_data, sr=samplerate, n_mfcc=13, n_fft=512, hop_length=160
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate(
            (normalize(mfcc), normalize(mfcc_delta), normalize(mfcc_delta2)), axis=0
        )

        seq_length = mfcc.shape[1] // 2
        sequences = np.concatenate([[seq_length], transcript]).astype(np.int32)
        mfcc_out = (
            overlap(mfcc.T.astype(np.float32))
            if self.overlap
            else mfcc.T.astype(np.float32)
        )
        return mfcc_out, sequences
