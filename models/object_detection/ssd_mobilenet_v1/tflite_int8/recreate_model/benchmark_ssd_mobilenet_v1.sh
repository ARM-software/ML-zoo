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
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -n annotations_trainval2017.zip

python3.7 -m venv venv

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

pip install tensorflow==2.5.0

python benchmark_model.py --path ssd_mobilenet_v1.tflite
