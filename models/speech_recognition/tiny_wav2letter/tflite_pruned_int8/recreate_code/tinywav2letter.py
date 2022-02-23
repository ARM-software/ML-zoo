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

"""Model definition for Tinywav2Letter."""
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
import numpy as np
from jiwer import wer

def get_metrics(metric):
    """Get metrics needed to compile wav2letter."""
    def ctc_preparation(tensor, y_predict):
        if len(y_predict.shape) == 4:
            y_predict = tf.squeeze(y_predict, axis=1)
        y_predict = tf.transpose(y_predict, (1, 0, 2))
        sequence_lengths, labels = tensor[:, 0], tensor[:, 1:]
        idx = tf.where(tf.not_equal(labels, 28))
        sparse_labels = tf.SparseTensor(
            idx, tf.gather_nd(labels, idx), tf.shape(labels, out_type=tf.int64)
        )
        return sparse_labels, sequence_lengths, y_predict

    def get_loss():
        """Calculate CTC loss."""
        def ctc_loss(y_true, y_predict):
            sparse_labels, logit_length, y_predict = ctc_preparation(y_true, y_predict)
            return tf.reduce_mean(
                ctc_ops.ctc_loss_v2(
                    labels=sparse_labels,
                    logits=y_predict,
                    label_length=None,
                    logit_length=logit_length,
                    blank_index=-1,
                )
            )
        return ctc_loss

    def get_ler():
        """Calculate CTC LER (Letter Error Rate)."""
        def ctc_ler(y_true, y_predict):
            sparse_labels, logit_length, y_predict = ctc_preparation(y_true, y_predict)
            decoded, log_probabilities = tf.nn.ctc_greedy_decoder(
                y_predict, tf.cast(logit_length, tf.int32), merge_repeated=True
            )
            return tf.reduce_mean(
                tf.edit_distance(
                    tf.cast(decoded[0], tf.int32), tf.cast(sparse_labels, tf.int32)
                )
            )
        return ctc_ler
    def get_wer():
        """Calculate CTC WER (Word Error Rate) only for batch size = 1."""

        def trans_int_to_string(trans_int):
            #create dictionary int -> string (0 -> a 1 -> b)
            string = ""
            alphabet = "abcdefghijklmnopqrstuvwxyz' @"
            alphabet_dict = {}
            count = 0
            for x in alphabet:
                alphabet_dict[count] = x
                count += 1
            for letter in trans_int:
                letter_np = np.array(letter).item(0)
                if letter_np != 28:
                    string += alphabet_dict[letter_np]
            return string

        def ctc_wer(y_true, y_predict):
            sparse_labels, logit_length, y_predict = ctc_preparation(y_true, y_predict)
            decoded, log_probabilities = tf.nn.ctc_greedy_decoder(
                y_predict, tf.cast(logit_length, tf.int32), merge_repeated=True
            )
            true_sentence = tf.cast(sparse_labels.values, tf.int32)
            return wer(str(trans_int_to_string(true_sentence)),str(trans_int_to_string(decoded[0].values)))
        return ctc_wer

    return {"loss": get_loss(), "ler": get_ler(), "wer": get_wer()}[metric]


def create_tinywav2letter(batch_size=1, no_stride_count=5, filters_small=100, filters_large_1=750, filters_large_2=750) -> tf.keras.models.Model:
    """Create and return Tinywav2Letter model"""
    layer = tf.keras.layers
    leaky_relu = layer.LeakyReLU([0.20000000298023224])
    MFCC_coeffs = 39
    input = layer.Input(shape=[None, MFCC_coeffs], batch_size=batch_size)
    # Reshape to prepare input for first layer
    x = layer.Reshape([1, -1, 39])(input)
    # One striding layer of output size [batch_size, max_time / 2, 250]
    x = layer.Conv2D(
        filters=250,
        kernel_size=[1, 48],
        padding="same",
        activation=None,
        strides=[1, 2],
    )(x)
    # Add non-linearity
    x = leaky_relu(x)
    # layers without striding of output size [batch_size, max_time / 2, 250]
    for i in range(0, no_stride_count):
        x = layer.Conv2D(
            filters=filters_small,
            kernel_size=[1, 7],
            padding="same",
            activation=None,
            strides=[1, 1],
        )(x)
        # Add non-linearity
        x = leaky_relu(x)
    # 1 layer with high kernel width and output size [batch_size, max_time / 2, 2000]
    x = layer.Conv2D(
        filters=filters_large_1,
        kernel_size=[1, 32],
        padding="same",
        activation=None,
        strides=[1, 1],
    )(x)
    # Add non-linearity
    x = leaky_relu(x)
    # 1 layer of output size [batch_size, max_time / 2, 2000]
    x = layer.Conv2D(
        filters=filters_large_2,
        kernel_size=[1, 1],
        padding="same",
        activation=None,
        strides=[1, 1],
    )(x)
    # Add non-linearity
    x = leaky_relu(x)
    # 1 layer of output size [batch_size, max_time / 2, num_classes]
    # We must not apply a non linearity in this last layer
    x = layer.Conv2D(
        filters=29,
        kernel_size=[1, 1],
        padding="same",
        activation=None,
        strides=[1, 1],
    )(x)
    return tf.keras.models.Model(inputs=[input], outputs=[x])
