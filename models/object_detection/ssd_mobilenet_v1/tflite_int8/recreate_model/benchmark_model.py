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
import tensorflow_datasets as tfds
import tensorflow as tf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

detections = []

# Yields the image pre-processed with it's class
def coco_generator(coco_dataset, input_size=(300, 300)):
    for item in coco_dataset:
        image = item['image']
        image = tf.image.resize(image, input_size)
        image = tf.expand_dims(image, 0)
        
        # MobileNet pre-processing
        image = (image / 255. - 0.5) * 2

        yield image, item['image/id'], item['image'].shape


def __convert_to_coco_bbox(b, input_size):
    # For COCO it is [x, y, width, height]
    # The bounding box b is of type: [ymin, xmin, ymax, xmax]
    x = b[1] * input_size[1]
    y = b[0] * input_size[0]
    width = (b[3] - b[1]) * input_size[1]
    height = (b[2] - b[0]) * input_size[0]

    return [x, y, width, height]


def process_output(output, image_id, image_size):
    detection_boxes, detection_classes, detection_scores, num_detections = output

    detections_in_image = []
    for i in range(int(num_detections[0])):
        detections_in_image.append(
            {
                'image_id': image_id.numpy(),
                'category_id': int(detection_classes[0, i]) + 1,
                'bbox': __convert_to_coco_bbox(detection_boxes[0, i], image_size), 
                'score': detection_scores[0, i]
            }
        )

    return detections_in_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark SSD MobileNet v1.')
    parser.add_argument('--path', type=str, help='Path to the model.')

    args = parser.parse_args()

    # Get the COCO 2017 validation set
    coco_dataset = tfds.load('coco/2017', split='validation')
    
    # Setup the TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_path=args.path)
    interpreter.allocate_tensors()

    input_node = interpreter.get_input_details()[0]
    input_t = input_node['index']
    output_t = [details['index'] for details in interpreter.get_output_details()]

    for data, data_id, image_shape in coco_generator(coco_dataset):
        # Quantize the input data
        scale = input_node["quantization_parameters"]["scales"]
        zero_point = input_node["quantization_parameters"]["zero_points"]

        data = data / scale
        data += zero_point

        numpy_data = tf.cast(data, tf.int8).numpy()
        interpreter.set_tensor(input_t, numpy_data)
        interpreter.invoke()

        output = [ interpreter.get_tensor(o) for o in output_t ]
        
        detection_outputs = process_output(output, data_id, (image_shape[0], image_shape[1]))
        detections += detection_outputs

    # Use the COCO API to measure the accuracy on the annotations
    coco_ground_truth = COCO('./annotations/instances_val2017.json')
    coco_results = coco_ground_truth.loadRes(detections)

    coco_eval = COCOeval(coco_ground_truth, coco_results, 'bbox')

    image_ids = [d['image_id'] for d in detections]
    coco_eval.params.imgIds = image_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
