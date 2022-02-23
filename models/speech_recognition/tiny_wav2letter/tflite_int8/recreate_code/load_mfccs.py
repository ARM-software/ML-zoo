# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
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

import tensorflow as tf
import os
import numpy as np

class MFCC_Loader:
    def __init__(self, full_size_data_dir:str, reduced_size_data_dir:str, fluent_speech_data_dir:str):
        """
            Args:
                data_dir: Absolute path to librispeech data folder
        """
        self.full_size_data_dir = full_size_data_dir
        self.reduced_size_data_dir = reduced_size_data_dir
        self.fluent_speech_data_dir = fluent_speech_data_dir
        self.seed = 0
        self.train = False
        self.batch_size = 32
        self.num_samples = 0
        self.input_files = []


    @staticmethod
    def _extract_features(example_proto):
        feature_description = {
            'mfcc_bytes': tf.io.FixedLenFeature([], tf.string),
            'sequence_bytes': tf.io.FixedLenFeature([], tf.string),
            }
        # Parse the input tf.train.Example proto using the dictionary above.
        serialized_tensor = tf.io.parse_single_example(example_proto, feature_description)

        mfcc_features = tf.io.parse_tensor(serialized_tensor['mfcc_bytes'], out_type = tf.float32)
        sequences = tf.io.parse_tensor(serialized_tensor['sequence_bytes'], out_type = tf.int32)

        return mfcc_features, sequences

    def full_training_set(self, batch_size=32, num_samples = -1):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = [
            os.path.join(self.full_size_data_dir, 'preprocessed/train-clean-100/train-clean-100.tfrecord'),
            os.path.join(self.full_size_data_dir, 'preprocessed/train-clean-360/train-clean-360.tfrecord')]
        
        self.train = True
        self.batch_size = batch_size
        self.num_samples = 132553
        return self.load_dataset(num_samples)

    def reduced_training_set(self, batch_size=32, num_samples = -1):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = os.path.join(self.reduced_size_data_dir, 'preprocessed/train-clean-5/train-clean-5.tfrecord')
        self.train = True
        self.batch_size = batch_size
        self.num_samples = 1519
        return self.load_dataset(num_samples)

    def full_validation_set(self, batch_size=32):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = os.path.join(self.full_size_data_dir, 'preprocessed/dev-clean/dev-clean.tfrecord')
        self.train = False
        self.batch_size = batch_size
        self.num_samples = 2703
        return self.load_dataset()

    def reduced_validation_set(self, batch_size=32):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = os.path.join(self.reduced_size_data_dir, 'preprocessed/dev-clean-2/dev-clean-2.tfrecord')
        self.train = False
        self.batch_size = batch_size
        self.num_samples = 1089
        return self.load_dataset()


    def evaluation_set(self, batch_size=32):
        """
        Args:
            batch_size: batch size required for the set
        """

        self.tfrecord_file = os.path.join(self.full_size_data_dir, 'preprocessed/test-clean/test-clean.tfrecord')
        self.train = False
        self.batch_size = batch_size
        self.num_samples = 2620
        return self.load_dataset()

    def fluent_speech_train_set(self, batch_size=32, num_samples = -1):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = os.path.join(self.fluent_speech_data_dir, 'preprocessed/train/train.tfrecord')
        
        self.train = True
        self.batch_size = batch_size
        self.num_samples = 23132
        return self.load_dataset(num_samples)

    def fluent_speech_validation_set(self, batch_size=32):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = os.path.join(self.fluent_speech_data_dir, 'preprocessed/dev/dev.tfrecord')
        self.train = False
        self.batch_size = batch_size
        self.num_samples = 3118
        return self.load_dataset()

    def fluent_speech_test_set(self, batch_size=32):
        """
        Args:
            batch_size: batch size required for the set
        """
        self.tfrecord_file = os.path.join(self.fluent_speech_data_dir, 'preprocessed/test/test.tfrecord')
        self.train = False
        self.batch_size = batch_size
        self.num_samples = 3793
        return self.load_dataset()

    def num_steps(self, batch):
        """
        Get the number of steps based on the given batch size and the number
        of samples.
        """
        return int(np.math.ceil(self.num_samples / batch))


    def load_dataset(self, num_samples = -1):

        # load the specified TF Record files
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)

        # parse the data, and take the desired number of samples
        dataset = dataset.map(self._extract_features, num_parallel_calls = tf.data.AUTOTUNE).take(num_samples)
        
        dataset = dataset.cache()
        
        # shuffle the training set
        if self.train:
            dataset = dataset.shuffle(buffer_size=max(self.batch_size * 2, 1024), seed=self.seed)

        MFCC_coeffs = 39
        blank_index = 28


        # Pad the dataset so that all the data is the same size
        dataset = dataset.padded_batch(
                self.batch_size,
                padded_shapes=(tf.TensorShape([None, MFCC_coeffs]), tf.TensorShape([None])),
                padding_values=(0.0, blank_index), drop_remainder=True
            )
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)
