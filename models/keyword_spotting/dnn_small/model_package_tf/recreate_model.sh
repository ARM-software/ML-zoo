#!/usr/bin/env bash
# Copyright (C) 2023 Arm Limited or its affiliates. All rights reserved.
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

ckpt_path=model_archive/model_source/weights/dnn_0.84_ckpt
train=false

# Parse command line args
while (( $# >= 1 )); do 
    case $1 in
    --ckpt)
       if [ "$2" ]; then
           ckpt_path=$2
           shift
       else
           printf 'ERROR: "--ckpt" requires a path to be supplied.\n'
           exit 1
       fi
       ;;
    --train) 
    	train=true
	break;;
    *) shift;
    esac;
done


# DNN Small training
if [ "$train" = true ]
then
python train.py --model_architecture dnn --model_size_info 144 144 144 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --learning_rate 0.0005,0.0001,0.00002 --how_many_training_steps 10000,10000,10000 --summaries_dir work/DNN/DNN_S/retrain_logs --train_dir work/DNN/DNN_S/training
fi

# Conversion to TFLite fp32
python convert_to_tflite.py --model_architecture dnn --model_size_info 144 144 144 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --checkpoint $ckpt_path --no-quantize

# Conversion to TFLite int8
python convert_to_tflite.py --model_architecture dnn --model_size_info 144 144 144 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --checkpoint $ckpt_path --inference_type int8

