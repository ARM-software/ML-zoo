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
"""Code for training the RNNoise model."""
import argparse
import logging

import tensorflow as tf
from tensorflow.keras import backend as K

from data import get_tf_dataset_from_h5
from model import rnnoise_model


def my_crossentropy(y_true, y_pred):
    """Cross entropy loss for vad output.
    0 if y_true label is 0.5 - meaning we aren't sure if speech is present.

    Ref:https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/training/rnn_train.py#L31"""
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)


def my_mask(y_true):
    """Used to mask off gain values if label indicates no audible signal.

    Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/training/rnn_train.py#L34
    """
    return K.minimum(y_true+1., 1.)


def msse(y_true, y_pred):
    """Masked mean sum of square errors."""
    return K.mean(my_mask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


def my_cost(y_true, y_pred):
    """Custom loss function for training noise reduction output.

    Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/training/rnn_train.py#L40."""
    return K.mean(my_mask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true)))
                                     + K.square(K.sqrt(y_pred) - K.sqrt(y_true))
                                     + 0.01*K.binary_crossentropy(y_pred, y_true)),
                  axis=-1)


def train():
    epochs = 120
    window_size = 2000
    batch_size = 32
    ckpt_path = './ckpts/{epoch}'
    logs_path = './logs'

    x_train = get_tf_dataset_from_h5(FLAGS.train_data_h5, window_size, batch_size)
    x_test = get_tf_dataset_from_h5(FLAGS.test_data_h5, window_size, batch_size)
    logging.info('Data loading complete.')

    model = rnnoise_model(window_size, None, is_training=True)

    model.compile(loss=[my_cost, my_crossentropy],
                  metrics=[msse],
                  optimizer='adam', loss_weights=[10, 0.5])

    # Callbacks for saving checkpoints and TensorBoard logging.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=logs_path)
        ]

    model.fit(x_train, validation_data=x_test, epochs=epochs, callbacks=callbacks)
    logging.info('Training complete.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_h5',
        type=str,
        required=True,
        help='Path to the training data in H5 format.'
    )
    parser.add_argument(
        '--test_data_h5',
        type=str,
        required=True,
        help='Path to the testing data in H5 format.'
    )
    FLAGS, _ = parser.parse_known_args()
    train()
