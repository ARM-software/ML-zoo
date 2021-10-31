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
"""Code for converting and quantizing trained RNNoise model to TFLite."""
import argparse
import logging

import tensorflow as tf
import numpy as np

from model import rnnoise_model_tflite
from data import get_tf_dataset_from_h5

NUM_CALIB = 1000


def collect_calibration_data(data_h5_path, ckpt_path):
    """Generate a calibration dataset for post-training quantization.
    
    We iteratively add GRU state outputs from the model inference to the calibration dataset.
    """
    ds_calib = get_tf_dataset_from_h5(data_h5_path, window_size=1, batch_size=1).take(NUM_CALIB)

    model = rnnoise_model_tflite(timesteps=1)
    model.load_weights(ckpt_path).expect_partial()

    # Initial GRU states are all zero.
    vad_gru_state = np.zeros((1, 24), dtype=np.float32)
    noise_gru_state = np.zeros((1, 48), dtype=np.float32)
    denoise_gru_state = np.zeros((1, 96), dtype=np.float32)
    
    calibration_set = []
    for data in ds_calib:
        calibration_set.append((data[0], vad_gru_state, noise_gru_state, denoise_gru_state))

        model_out, vad_out, vad_gru_state, noise_gru_state, denoise_gru_state = model.predict([data[0],
                                                                                               vad_gru_state,
                                                                                               noise_gru_state,
                                                                                               denoise_gru_state])
        # Remove extra dimension from model outputs.
        vad_gru_state = np.squeeze(vad_gru_state, axis=1)
        noise_gru_state = np.squeeze(noise_gru_state, axis=1)
        denoise_gru_state = np.squeeze(denoise_gru_state, axis=1)

    return calibration_set


def post_training_quantize(ckpt_path):
    """Generate fully quantized int8 1 step RNNoise model."""
    rnnoise_model = rnnoise_model_tflite(timesteps=1)
    rnnoise_model.load_weights(ckpt_path).expect_partial()

    converter = tf.lite.TFLiteConverter.from_keras_model(rnnoise_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    def rep_dataset():
        """RNNoise Representative dataset generator for quantizing."""
        data_s = collect_calibration_data(FLAGS.h5_path, FLAGS.ckpt_path)

        for data in data_s:
            input_tensor, vad_gru, noise_gru, denoise_gru = data

            yield [input_tensor, vad_gru, noise_gru, denoise_gru]

    converter.representative_dataset = rep_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    with open('rnnoise_1_step_int8.tflite', 'wb') as f:
        f.write(tflite_quant_model)
    logging.info('INT8 TFLite model generated.')


def tflite_convert(ckpt_path):
    """Generate fp32 1 step RNNoise model."""
    rnnoise_model = rnnoise_model_tflite(timesteps=1)
    rnnoise_model.load_weights(ckpt_path).expect_partial()

    converter = tf.lite.TFLiteConverter.from_keras_model(rnnoise_model)
    tflite_quant_model = converter.convert()

    with open('rnnoise_1_step_fp32.tflite', 'wb') as f:
        f.write(tflite_quant_model)
    logging.info('FP32 TFLite model generated.')


def main():
    tflite_convert(FLAGS.ckpt_path)
    post_training_quantize(FLAGS.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to the trained RNNoise ckpt files.'
    )
    parser.add_argument(
        '--h5_path',
        type=str,
        required=True,
        help='Path to a h5 file to use for generating the calibration dataset.'
    )
    FLAGS, _ = parser.parse_known_args()
    main()
