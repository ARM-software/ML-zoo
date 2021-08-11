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


import tensorflow as tf
import os

class TFRecordWriter:

    def __init__(self, data_directory):
        self.data_directory = data_directory

        for item in os.listdir(self.data_directory):
            if item[-3:] == 'txt':
                text_file = item
            elif '.' not in item:
                self.image_directory = item
        
        self.image_directory = 'validation_images'

        self.class_label_dict = self.build_class_label_dictionary(os.path.join(data_directory, text_file))

    @staticmethod
    def extract_image_id_and_class_label(row):
        example_file_name = "ILSVRC2012_val_00000000.JPEG"
        image_id = row[:len(example_file_name)]
        class_label = row[len(example_file_name) + 1:-1]

        return image_id, class_label

    def build_class_label_dictionary(self, text_file_path):
        class_label_dict = {}
        with open(text_file_path, 'r') as f:
            for row in f:
                image, class_label = self.extract_image_id_and_class_label(row)
                class_label_dict[image] = int(class_label)
            
        return class_label_dict

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _create_feature(self, image_binary, class_label):
            """
            Creates a tf.train.Example message ready to be written to a file.
            """
            # Create a dictionary mapping the feature name to the tf.train.Example-compatible
            # data type.

            feature = {
                'image/encoded': self._bytes_feature(image_binary),
                'image/class/label': self._int64_feature(class_label),
            }

            # Create a Features message using tf.train.Example.
            return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tf_record(self, tf_record_filename):
        """
        Loads each image/class label in the dictionary and writes them to a TFRecord file
        
        tf_record_filename: String, the name of the file to write the data to.

        """
        
        with tf.io.TFRecordWriter(os.path.join(self.data_directory, tf_record_filename)) as writer:
            for image_file_name, class_label in self.class_label_dict.items():
                image_path = self.data_directory + '/' + self.image_directory + '/' + image_file_name
                with open(image_path, 'rb') as image:
                    writer.write(self._create_feature(image.read(), class_label).SerializeToString())

    
if __name__=="__main__":
    writer = TFRecordWriter('data/validation_data')
    writer.write_tf_record('validation-dataset.tfrecord')
