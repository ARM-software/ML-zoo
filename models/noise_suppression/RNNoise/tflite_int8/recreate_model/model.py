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
"""Code for defining the RNNoise model."""
import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class WeightClip(Constraint):
    """tf.keras Constraint for doing weight clipping."""
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max)

    def get_config(self):
        return {'min_clip_value': self.min,
                'max_clip_value': self.max}


def rnnoise_model(timesteps, batch_size, is_training):
    """RNNoise model definition for training and evaluating, not for TFLite deployment."""
    constraint = WeightClip(-0.499, 0.499)
    reg = tf.keras.regularizers.L2(0.000001)

    stateful = False if is_training else True

    main_input = tf.keras.Input(shape=(timesteps, 42), batch_size=batch_size, name='main_input')

    fc1 = tf.keras.layers.Dense(24, activation='tanh', kernel_constraint=constraint,
                                bias_constraint=constraint, name='fc1')(main_input)

    vad_gru_out = tf.keras.layers.GRU(units=24, return_sequences=True, unroll=False, kernel_regularizer=reg,
                                      recurrent_regularizer=reg, kernel_constraint=constraint,
                                      recurrent_constraint=constraint, bias_constraint=constraint,
                                      stateful=stateful, name='vad_gru')(fc1)

    vad_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_constraint=constraint,
                                    bias_constraint=constraint, name='vad_output')(vad_gru_out)

    noise_input = tf.keras.layers.concatenate([fc1, vad_gru_out, main_input])

    noise_out = tf.keras.layers.GRU(units=48, activation='relu', return_sequences=True, unroll=False,
                                    kernel_regularizer=reg, recurrent_regularizer=reg, kernel_constraint=constraint,
                                    recurrent_constraint=constraint, bias_constraint=constraint,
                                    stateful=stateful, name='noise_gru')(noise_input)

    denoise_input = tf.keras.layers.concatenate([vad_gru_out, noise_out, main_input])

    denoise_gru_out = tf.keras.layers.GRU(units=96, return_sequences=True, unroll=False, kernel_regularizer=reg,
                                          recurrent_regularizer=reg, kernel_constraint=constraint,
                                          recurrent_constraint=constraint, bias_constraint=constraint,
                                          stateful=stateful, name='denoise_gru')(denoise_input)

    denoise_out = tf.keras.layers.Dense(units=22, activation='sigmoid', kernel_constraint=constraint,
                                        bias_constraint=constraint, name='denoise_output')(denoise_gru_out)

    model = tf.keras.Model([main_input],
                           [denoise_out, vad_out])
    return model


def rnnoise_model_tflite(timesteps):
    """Unrolled model definition of RNNoise specifically for producing a quantized TFLite model.

    Input and output GRU states are exposed as well.
    """
    main_input = tf.keras.Input(shape=(timesteps, 42), batch_size=1, name='main_input')
    vad_gru_state = tf.keras.Input(shape=24, batch_size=1, name='vad_gru_prev_state')
    noise_gru_state = tf.keras.Input(shape=48, batch_size=1, name='noise_gru_prev_state')
    denoise_gru_state = tf.keras.Input(shape=96, batch_size=1, name='denoise_gru_prev_state')

    fc1 = tf.keras.layers.Dense(24, activation='tanh', name='fc1')(main_input)

    vad_gru_out = tf.keras.layers.GRU(units=24, return_sequences=True, unroll=True,
                                      name='vad_gru')(fc1, initial_state=vad_gru_state)

    vad_out = tf.keras.layers.Dense(1, activation='sigmoid', name='vad_output')(vad_gru_out)

    noise_input = tf.keras.layers.concatenate([fc1, vad_gru_out, main_input])

    noise_gru_out = tf.keras.layers.GRU(units=48, activation='relu', return_sequences=True, unroll=True,
                                        name='noise_gru')(noise_input, initial_state=noise_gru_state)

    denoise_input = tf.keras.layers.concatenate([vad_gru_out, noise_gru_out, main_input])

    denoise_gru_out = tf.keras.layers.GRU(units=96, return_sequences=True, unroll=True,
                                          name='denoise_gru')(denoise_input, initial_state=denoise_gru_state)

    denoise_out = tf.keras.layers.Dense(units=22, activation='sigmoid', name='denoise_output')(denoise_gru_out)

    model = tf.keras.Model([main_input, vad_gru_state, noise_gru_state, denoise_gru_state],
                           [denoise_out, vad_out, vad_gru_out, noise_gru_out, denoise_gru_out])
    return model
