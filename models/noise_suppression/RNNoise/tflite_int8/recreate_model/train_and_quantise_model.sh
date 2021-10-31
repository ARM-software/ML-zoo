#!/usr/bin/env bash
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

python3 -m venv venv

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python train.py --train_data_h5=./train.h5 --test_data_h5=./test.h5
python convert.py --ckpt_path=./ckpts/120 --h5_path=./test.h5
