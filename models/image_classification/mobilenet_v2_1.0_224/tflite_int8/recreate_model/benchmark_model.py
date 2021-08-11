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


import time
import argparse

import numpy as np
import tensorflow as tf

from load_data import DatasetLoader

def benchmark_tflite_model(tflite_model_path, data_dir = 'data', validation_set_size=-1):
    validation_dataset = DatasetLoader(data_dir).load_data(validation_set_size)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_node = interpreter.get_input_details()[0]
    input_index = input_node["index"]

    output_node = interpreter.get_output_details()[0]
    output_index = output_node["index"]

    input_scale = input_node['quantization_parameters']['scales']
    input_zero_point = input_node['quantization_parameters']['zero_points']

    # Run predictions on every image in the "test" dataset.
    predictions = []
    true_class_labels = []

    start = time.time()
    for i, (test_image, class_label) in enumerate(validation_dataset):
        
        test_image = tf.expand_dims(test_image, axis=0)
        test_image = tf.math.divide(test_image, input_scale)
        test_image = tf.math.add(test_image, input_zero_point)
        test_image = tf.cast(test_image, tf.int8)

        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        predicted_class = np.argmax(output()[0])

        predictions.append(predicted_class)
        true_class_labels.append(class_label)

        if i % 100 == 99:
            predictions_array= np.array(predictions)
            true_class_labels_array = np.array(true_class_labels)
            accuracy = (predictions_array == true_class_labels_array).mean()
            stop = time.time()
            print(  f'Evaluated on {i+1} results so far: '
                    f'Accuracy: {accuracy}. '
                    f'Time taken: {stop-start} seconds.'
                    )

        # Compare prediction results with ground truth labels to calculate accuracy.

    predictions_array= np.array(predictions)
    true_class_labels_array = np.array(true_class_labels)
    accuracy = (predictions_array == true_class_labels_array).mean()
        
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark MobileNet v2 INT8.')
    parser.add_argument('--path', type=str, help='Path to the model.', required = True)

    args = parser.parse_args()

    accuracy = benchmark_tflite_model(args.path)
