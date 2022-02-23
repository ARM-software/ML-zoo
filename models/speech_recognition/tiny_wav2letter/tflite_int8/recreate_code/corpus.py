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

import os
import tarfile
import urllib.request


class SpeechCorpusProvider:
    """
    Ensures the availability and downloads the speech corpus if necessary
    """

    DATA_SOURCE = {
        "librispeech_full_size": 
            {"DATA_SETS":
                {
                    ('train', 'train-clean-100'),
                    ('train', 'train-clean-360'),
                    ('val', 'dev-clean')
                },
            "BASE_URL": 'http://www.openslr.org/resources/12/'
            },
        "librispeech_reduced_size": 
            {"DATA_SETS":
                {
                    ('train', 'train-clean-5'),
                    ('val', 'dev-clean-2')
                },
            "BASE_URL": 'http://www.openslr.org/resources/31/'
            }
        }

    SET_FILE_EXTENSION = '.tar.gz'
    TAR_ROOT = 'LibriSpeech/'

    def __init__(self, data_directory):
        """
        Creates a new SpeechCorpusProvider with the root directory `data_directory`.
        The speech corpus is downloaded and extracted into sub-directories.

        Args:
            data_directory: the root directory to use, e.g. data/
        """

        self._data_directory = data_directory
        self._make_dir_if_not_exists(data_directory)
        self.data_sets = SpeechCorpusProvider.DATA_SOURCE[data_directory]['DATA_SETS']
        self.base_url = SpeechCorpusProvider.DATA_SOURCE[data_directory]['BASE_URL']

    @staticmethod
    def _make_dir_if_not_exists(directory):
        """
        Helper function to create a directory if it doesn't exist.

        Args:
            directory: directory to create
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

    def _download_if_not_exists(self, remote_file_name):
        """
        Downloads the given `remote_file_name` if not yet stored in the `data_directory`

        Args:
            remote_file_name: the file to download

        Returns: path to downloaded file
        """

        path = os.path.join(self._data_directory, remote_file_name)
        if not os.path.exists(path):
            print('Downloading {}...'.format(remote_file_name))
            urllib.request.urlretrieve(self.base_url + remote_file_name, path)
        return path

    @staticmethod
    def _extract_from_to(tar_file_name, source, target_directory):
        """
        Extract all necessary files `source` from `tar_file_name` into `target_directory`

        Args:
            tar_file_name: the tar file to extract from
            source: the directory in the root to extract
            target_directory: the directory to store the files in
        """

        print('Extracting {}...'.format(tar_file_name))
        with tarfile.open(tar_file_name, 'r:gz') as tar:
            source_members = [
                tarinfo for tarinfo in tar.getmembers()
                if tarinfo.name.startswith(SpeechCorpusProvider.TAR_ROOT + source)
                ]
            for member in source_members:
                # Extract without prefix
                member.name = member.name.replace(SpeechCorpusProvider.TAR_ROOT, '')
            tar.extractall(target_directory, source_members)

    def _is_ready(self):
        """
        Returns whether all `data_sets` are downloaded and extracted

        Args:
            data_sets: list of the datasets to ensure

        Returns: bool, is ready to use

        """

        data_set_paths = [os.path.join(set_type, set_name)
                            for set_type, set_name in self.data_sets]
        
        return all([os.path.exists(os.path.join(self._data_directory, data_set)) 
            for data_set in data_set_paths])

    def _download(self):
        """
        Download the given `data_sets`

        Args:
            data_sets: a list of the datasets to download
        """

        for data_set_type, data_set_name in self.data_sets:
            remote_file = data_set_name + SpeechCorpusProvider.SET_FILE_EXTENSION
            self._download_if_not_exists(remote_file)

    def _extract(self):
        """
        Extract all necessary files from the given `data_sets`
        """

        for data_set_type, data_set_name in self.data_sets:
            local_file = os.path.join(self._data_directory, data_set_name + SpeechCorpusProvider.SET_FILE_EXTENSION)
            target_directory = self._data_directory
            self._extract_from_to(local_file, data_set_name, target_directory)

    def ensure_availability(self):
        """
        Ensure that all datasets are downloaded and extracted. If this is not the case,
        the download and extraction is initated.
        """

        if not self._is_ready():
            self._download()
            self._extract()


if __name__=="__main__":
    full_size_corpus = SpeechCorpusProvider("librispeech_full_size")
    full_size_corpus.ensure_availability()

    reduced_size_corpus = SpeechCorpusProvider("librispeech_reduced_size")
    reduced_size_corpus.ensure_availability()


