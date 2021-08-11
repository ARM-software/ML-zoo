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


import os
import tensorflow as tf

from load_data import DatasetLoader


class Quantisation:

    def __init__(self, data_dir, output_dir = 'tflite', validation_set_size = 100):
        self.output_directory = output_dir
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.validation_dataset = DatasetLoader(data_dir).load_data(validation_set_size)

    def convert_to_tflite(self):

        def representative_dataset():
            for image, _ in self.validation_dataset.batch(1):
                yield [tf.dtypes.cast(image, tf.float32)]

        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file="mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb",
                input_arrays=['input'],
                output_arrays=['MobilenetV2/Predictions/Reshape_1'],
                input_shapes={'input': [1, 224, 224, 3]}
            )

        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_converter = True
        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]

        model = converter.convert()

        with open(self.output_directory + '/mobilenet_v2_1.0_224_INT8.tflite', 'wb') as f:
            f.write(model)


if __name__ == "__main__":
    quantise = Quantisation('data')
    quantise.convert_to_tflite()
