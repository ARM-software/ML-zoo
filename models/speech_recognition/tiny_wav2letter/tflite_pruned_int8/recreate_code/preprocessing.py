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

import fnmatch
import os
import librosa

import numpy as np
import tensorflow as tf
from corpus import SpeechCorpusProvider
from preprocessing_fluent_speech_commands import preprocess_fluent_sppech
from preprocessing_convert_to_flac import convert_to_flac
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def normalize(values):
    """
    Normalize values to mean 0 and std 1
    """
    return (values - np.mean(values)) / np.std(values)


def iglob_recursive(directory, file_pattern):
    """
    Recursively search for `file_pattern` in `directory`

    Args:
        directory: the directory to search in
        file_pattern: the file pattern to match (wildcard compatible)

    Returns: 
        iterator for found files

    """
    for root, dir_names, file_names in os.walk(directory):
        for filename in fnmatch.filter(file_names, file_pattern):
            yield os.path.join(root, filename)


class SpeechCorpusReader:
    """
    Reads preprocessed speech corpus to be used by the NN
    """
    def __init__(self, data_directory):
        """
        Create SpeechCorpusReader and read samples from `data_directory`

        Args:
        data_directory: the directory to use
        """
        self._data_directory = data_directory
        self._transcript_dict = self._build_transcript()

    @staticmethod
    def _get_transcript_entries(transcript_directory):
        """
        Iterate over all transcript lines and yield splitted entries

        Args:
        transcript_directory: open all transcript files in this directory and extract their contents

        Returns: Iterator for all entries in the form (id, sentence)

        """
        transcript_files = iglob_recursive(transcript_directory, '*.trans.txt')
        for transcript_file in transcript_files:
            with open(transcript_file, 'r') as f:
                for line in f:
                    # Strip included new line symbol
                    line = line.rstrip('\n')

                    # Each line is in the form
                    # 00-000000-0000 WORD1 WORD2 ...
                    splitted = line.split(' ', 1)
                    yield splitted

    def _build_transcript(self):
        """
        Builds a transcript from transcript files, mapping from audio-id to a list of vocabulary ids

        Returns: the created transcript
        """
        alphabet = "abcdefghijklmnopqrstuvwxyz' @"
        alphabet_dict = {c: ind for (ind, c) in enumerate(alphabet)}
        
        # Create the transcript dictionary
        transcript_dict = dict()
        for splitted in self._get_transcript_entries(self._data_directory):
            transcript_dict[splitted[0]] = [alphabet_dict[letter] for letter in splitted[1].lower()]

        return transcript_dict

    @classmethod
    def _extract_audio_id(cls, audio_file):
        file_name = os.path.basename(audio_file)
        audio_id = os.path.splitext(file_name)[0]

        return audio_id

    @staticmethod
    def transform_audio_to_mfcc(audio_file, transcript, n_mfcc=13, n_fft=512, hop_length=160):
        """
        Calculate mfcc coefficients from the given raw audio data

        Args:
            audio_file: .flac audio file
            n_mfcc: the number of coefficients to generate
            n_fft: the window size of the fft
            hop_length: the hop length for the window

        Returns: 
            the mfcc coefficients in the form [time, coefficients]
            sequences: the transcript of the audio file

        """

        audio_data, sample_rate = librosa.load(audio_file, sr=16000)

        mfcc = librosa.feature.mfcc(audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # add derivatives and normalize
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate((normalize(mfcc), normalize(mfcc_delta), normalize(mfcc_delta2)), axis=0) #mfcc is now 13+13+13=39 (according to our input shpe)

        seq_length = mfcc.shape[1] // 2

        sequences = np.concatenate([[seq_length], transcript]).astype(np.int32)
        mfcc_out = mfcc.T.astype(np.float32)
    
        return mfcc_out, sequences

    @staticmethod
    def _create_feature(mfcc_bytes, sequence_bytes):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.

        feature = {
            'mfcc_bytes': _bytes_feature(mfcc_bytes),
            'sequence_bytes': _bytes_feature(sequence_bytes),
        }

        # Create a Features message using tf.train.Example.
        return tf.train.Example(features=tf.train.Features(feature=feature))


    def _get_directory(self, sub_directory):
        preprocess_directory = 'preprocessed'

        directory = self._data_directory + '/' + preprocess_directory + '/' + sub_directory

        return directory


    def process_data(self, directory):
        """
        Read audio files from `directory` and store the preprocessed version in preprocessed/`directory`

        Args:
        directory: the sub-directory to read from

        """
        # create a list of all the .flac files
        audio_files = list(iglob_recursive(self._data_directory + '/' + directory , '*.flac'))

        out_directory = self._get_directory(directory)

        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

            # the file the TFRecord will be written to
            filename = out_directory + f'/{directory}.tfrecord'
            with tf.io.TFRecordWriter(filename) as writer:
                for audio_file in audio_files:
                    if os.path.getsize(audio_file) >= 13885: #small files are not suported
                        audio_id = self._extract_audio_id(audio_file)

                        # identify the transcript corresponding to the audio file
                        transcript = self._transcript_dict[audio_id]

                        # convert the audio to MFCCs
                        mfcc_feature, sequences = self.transform_audio_to_mfcc(audio_file, transcript)

                        # Serialise the MFCCs and transcript
                        mfcc_bytes = tf.io.serialize_tensor(mfcc_feature)
                        sequence_bytes = tf.io.serialize_tensor(sequences)

                        # Write to the TFRecord file
                        writer.write(self._create_feature(mfcc_bytes, sequence_bytes).SerializeToString())

        else:
            print('Processed data already exists')


class Preprocessing:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def run(self):
        # Download the raw data
        corpus = SpeechCorpusProvider(self.data_dir)
        corpus.ensure_availability()
        
        corpus_reader = SpeechCorpusReader(self.data_dir)

        # initialise the datasets
        data_sets = [data_set[1] for data_set in corpus.data_sets]

        for data_set in data_sets:
            print(f"Preprocessing {data_set} data")
            corpus_reader.process_data(data_set)

        print('Preprocessing Complete')
    def run_without_download(self):
        corpus_reader = SpeechCorpusReader(self.data_dir)
        corpus_reader.process_data('dev')
        corpus_reader.process_data('train')
        corpus_reader.process_data('test')
if __name__=="__main__":
    reduced_preprocessing = Preprocessing('librispeech_reduced_size')
    reduced_preprocessing.run()

    full_preprocessing = Preprocessing('librispeech_full_size')
    full_preprocessing.run()
    
    preprocess_fluent_sppech()
    convert_to_flac()
    fluent_speech_preprocessing = Preprocessing('fluent_speech_commands_dataset')  #please note this is a license data set
    fluent_speech_preprocessing.run_without_download()



