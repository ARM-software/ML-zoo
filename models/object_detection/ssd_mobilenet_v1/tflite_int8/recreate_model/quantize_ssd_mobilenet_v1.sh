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

#!/usr/bin/env bash
python3.7 -m venv venv

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

git clone https://github.com/tensorflow/models.git

pushd models/research
export PYTHONPATH=`pwd`:`pwd`/slim:$PYTHONPATH
protoc object_detection/protos/*.proto --python_out=.
popd

pushd models/
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz

mkdir ssd_tflite
python research/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path ssd_mobilenet_v1_coco_2018_01_28/pipeline.config \
    --trained_checkpoint_prefix ssd_mobilenet_v1_coco_2018_01_28/model.ckpt \
    --output_directory ssd_tflite/ \
    --max_detections=100 \
    --add_postprocessing_op=true

mv ssd_tflite/ ..
popd

pip install tensorflow==2.5.0
python quantize_model.py
