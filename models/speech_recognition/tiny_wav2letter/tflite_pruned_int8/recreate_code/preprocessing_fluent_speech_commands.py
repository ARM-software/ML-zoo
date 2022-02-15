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

from csv import reader
import os
from shutil import copyfile
from pathlib import Path
import re
def preprocess(flag):
   if flag == 'train':
     path_csv = 'fluent_speech_commands_dataset/data/train_data.csv'
     path_dir = 'fluent_speech_commands_dataset/train/'
   elif flag == 'dev':
       path_csv = 'fluent_speech_commands_dataset/data/valid_data.csv'
       path_dir = 'fluent_speech_commands_dataset/dev/'
   else:
       path_csv = 'fluent_speech_commands_dataset/data/test_data.csv'
       path_dir = 'fluent_speech_commands_dataset/test/'
   with open(path_csv, 'r') as read_obj:
       csv_reader = reader(read_obj)
       if not os.path.exists(path_dir):
          os.makedirs(path_dir)
       with open(path_dir + flag + '.trans.txt', 'w') as write_obj:
          for row in csv_reader:
             print(row)
             if(row[1] == 'path'):
                continue
             head, file_name = os.path.split(row[1])
             copyfile('fluent_speech_commands_dataset/' + row[1],path_dir + file_name)
             text = row[3]
             text = text.upper()
             text = re.sub('[^a-zA-Z \']+', '', text) #remove all other chars
             write_obj.write(Path(file_name).stem + " " + text + '\n')
def preprocess_fluent_sppech():
    preprocess('train')
    preprocess('dev')
    preprocess('test')
