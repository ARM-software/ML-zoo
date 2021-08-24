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

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(coco_dataset, input_size=(300, 300)):
    def representative_dataset_gen():
        for example in coco_dataset.take(10000):
            image = tf.image.resize(example['image'], input_size)
            image = tf.expand_dims(image, 0)
            image = (2.0 / 255.0) * image - 1.0

            yield [image.numpy()]

    return representative_dataset_gen

if __name__ == "__main__":
    # Get the COCO 2017 dataset
    coco_dataset = tfds.load('coco/2017', split='train[:10%]')

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file='ssd_tflite/tflite_graph.pb',
        input_arrays=['normalized_input_image_tensor'],
        output_arrays=[
            'TFLite_Detection_PostProcess:0',
            'TFLite_Detection_PostProcess:1',
            'TFLite_Detection_PostProcess:2',
            'TFLite_Detection_PostProcess:3',
        ],
        input_shapes={
            'normalized_input_image_tensor': [1, 300, 300, 3]
        }
    )

    # Configure the TF Lite Converter
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.representative_dataset = get_dataset(coco_dataset, (300, 300)) 

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                           tf.lite.OpsSet.TFLITE_BUILTINS]

    model = converter.convert()

    with open('ssd_mobilenet_v1.tflite', 'wb') as f:
        f.write(model)
