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

from functools import partial
from corpus import ImageCorpusExtractor

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ImageCorpusReader:

    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)

    def __init__(self, data_directory, validation_dir):
        self.data_directory = data_directory
        self.validation_filenames = tf.io.gfile.glob(data_directory + '/' + validation_dir + "/validation-*")

    @staticmethod
    def decode_image(image):
        """ Decode the binary string to tensor, and crop the image.
        """
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

        return image

    def read_tfrecord(self, example, labeled):
        """ Define the TFRecord format, amd how each feature shall be processed.
        """
        tfrecord_format = (
            {
                "image/encoded": tf.io.FixedLenFeature([], tf.string),
                "image/class/label": tf.io.FixedLenFeature([], tf.int64),
            }
            if labeled
            else {"image/encoded": tf.io.FixedLenFeature([], tf.string),}
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self.decode_image(example["image/encoded"])
        if labeled:
            label = tf.cast(example["image/class/label"], tf.int32)
            
            return image, label
        return image

    @staticmethod
    def _preprocessing(image, class_label):
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, class_label

    def _load_dataset(self, filenames, labeled=True):
        """ Load the TFRecord as a dataset.
        """
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        
        # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
        dataset = dataset.map(
            partial(self.read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(self._preprocessing, num_parallel_calls=AUTOTUNE)
        
        # We need to crop the image, as the image saved in the TFRecord still keeps the original image's resolution.
        dataset = dataset.map(lambda x, y :  (tf.image.resize(x, (224, 224)), y))
        return dataset

    def prepare_dataset(self, dataset_size, labeled=True):
        """ Return the dataset given the filenames of TFRecords.
        """
        dataset = self._load_dataset(self.validation_filenames)
        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset.take(dataset_size)

class DatasetLoader:
    def __init__(self, data_dir):
        self.data_directory = data_dir
    
    def load_data(self, dataset_size):
        corpus = ImageCorpusExtractor(self.data_directory)
        corpus.ensure_availability()

        corpus_reader = ImageCorpusReader(self.data_directory, corpus.VALIDATION_DIR)
        validation_dataset = corpus_reader.prepare_dataset(dataset_size)
        return validation_dataset

if __name__ == "__main__":
    data_loader = DatasetLoader('data')
    validation_dataset = data_loader.load_data(dataset_size=-1)
    print('Extracting and Loading Complete')
