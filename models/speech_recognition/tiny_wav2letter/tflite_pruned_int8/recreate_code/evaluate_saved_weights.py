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

import argparse

import tensorflow as tf

from tinywav2letter import get_metrics, create_tinywav2letter
from train_model import get_data

def evaluate_saved_weights(args, pruned = False):

    model = create_tinywav2letter(batch_size = args.batch_size)

    model.load_weights('weights/tiny_wav2letter' + pruned * "_pruned" + '_weights.h5')

    opt = tf.keras.optimizers.Adam()
    model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt)

    (reduced_validation_data, reduced_validation_num_steps) = get_data(args, "val_reduced_size", args.batch_size)

    model.evaluate(reduced_validation_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        required=False,
        default=32,
        help="batch size wanted when creating model",
    )

    args = parser.parse_args()
    evaluate_saved_weights(args)

