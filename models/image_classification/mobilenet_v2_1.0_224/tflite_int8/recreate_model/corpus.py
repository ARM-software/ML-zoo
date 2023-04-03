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


import os
import tarfile

class ImageCorpusExtractor:

    VALIDATION_ZIP = 'validation_tf_records.tar.gz'
    VALIDATION_DIR = 'validation_data'


    def __init__(self, data_directory):
        self.data_directory = data_directory
        self._make_dir_if_not_exists(os.path.join(data_directory, ImageCorpusExtractor.VALIDATION_DIR))

    @staticmethod
    def _make_dir_if_not_exists(directory):
        """
        Helper function to create a directory if it doesn't exist.

        Args:
            directory: directory to create
        """

        if not os.path.exists(directory):
            os.makedirs(directory)


    def extract_zip_file(self, tar_file_path, target_directory):
            """
            Extract all necessary files `source` from `tar_file_name` into `target_directory`

            Args:
                tar_file_name: the tar file to extract from
                source: the directory in the root to extract
                target_directory: the directory to store the files in
            """
            

            print('Extracting {}...'.format(tar_file_path))
            with tarfile.open(tar_file_path) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, target_directory)

    def extract_dataset(self, zip_file, out_directory):
        zip_path = os.path.join(self.data_directory, zip_file)
        target_directory = os.path.join(self.data_directory, out_directory)

        if os.path.isfile(zip_path):
            if len(os.listdir(target_directory))==0:
                self.extract_zip_file(zip_path, target_directory)
            else:
                print('Zip file extracted')
        else:
            print(f'{zip_file} does not exist')
        
    def ensure_availability(self):
        self.extract_dataset(ImageCorpusExtractor.VALIDATION_ZIP, ImageCorpusExtractor.VALIDATION_DIR)
    

if __name__ == "__main__":
    corpus = ImageCorpusExtractor('data')
    corpus.ensure_availability()
