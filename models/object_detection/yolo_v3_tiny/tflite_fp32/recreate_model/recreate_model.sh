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

#!/usr/bin/env bash

python3.6 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

git clone https://github.com/mystic123/tensorflow-yolo-v3
pushd tensorflow-yolo-v3

wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3-tiny.weights

python convert_weights_pb.py --class_names coco.names --weights_file yolov3-tiny.weights --data_format NHWC --tiny

pip install tensorflow==1.15.0

tflite_convert --graph_def_file=frozen_darknet_yolov3_model.pb --output_file=yolo_v3_tiny_darknet_fp32.tflite --input_shapes=1,416,416,3 --input_arrays=inputs --output_arrays=output_boxes
mv yolo_v3_tiny_darknet_fp32.tflite ..

popd
