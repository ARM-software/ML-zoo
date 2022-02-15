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

from queue import Queue
import logging
import os
from threading import Thread
import audiotools
from audiotools.wav import InvalidWave

class W2F:

    logger = ''

    def __init__(self):
        global logger
        # create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create a file handler
        handler = logging.FileHandler('converter.log')
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

    def convert(self,path):
        global logger
        file_queue = Queue()
        num_converter_threads = 5

        # collect files to be converted
        for root, dirs, files in os.walk(path):

            for file in files:
                if file.endswith(".wav"):
                    file_wav = os.path.join(root, file)
                    file_flac = file_wav.replace(".wav", ".flac")

                    if (os.path.exists(file_flac)):
                        logger.debug(''.join(["File ",file_flac, " already exists."]))
                    else:
                        file_queue.put(file_wav)

        logger.info("Start converting:  %s files", str(file_queue.qsize()))

        # Set up some threads to convert files
        for i in range(num_converter_threads):
            worker = Thread(target=self.process, args=(file_queue,))
            worker.setDaemon(True)
            worker.start()

        file_queue.join()

    def process(self, q):
        """This is the worker thread function.
        It processes files in the queue one after
        another.  These daemon threads go into an
        infinite loop, and only exit when
        the main thread ends.
        """
        while True:
            global logger
            compression_quality = '0' #min compression
            file_wav = q.get()
            file_flac = file_wav.replace(".wav", ".flac")

            try:
                audiotools.open(file_wav).convert(file_flac,audiotools.FlacAudio, compression_quality)
                logger.info(''.join(["Converted ", file_wav, " to: ", file_flac]))
                q.task_done()
            except InvalidWave:
                logger.error(''.join(["Failed to open file ", file_wav, " to: ", file_flac," failed."]), exc_info=True)

def convert_to_flac():
    reduced_preprocessing = W2F()
    reduced_preprocessing.convert("fluent_speech_commands_dataset/train/")
    reduced_preprocessing.convert("fluent_speech_commands_dataset/dev/")
    reduced_preprocessing.convert("fluent_speech_commands_dataset/test/")
    print('')


