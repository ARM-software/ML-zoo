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

"""Wav2letter optimization and evaluation script"""
import argparse
import datetime
import multiprocessing
import os
import pathlib

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm
import numpy as np

from wav2letter import create_wav2letter, get_metrics
from librispeech_mfcc import LibriSpeechMfcc


def log(std):
    """Log the given string to the standard output."""
    print("******* {}".format(std), flush=True)


def create_directories(paths):
    """Directory creation"""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_data(dataset_type, overlap=False):
    """Returns particular training, validation and evaluation dataset."""
    dataset = LibriSpeechMfcc(args.data_dir)

    return {"train": [dataset.training_set(batch_size=args.batch_size, overlap=overlap),
                      dataset.num_steps(batch=args.batch_size)],
            "val":   [dataset.validation_set(batch_size=args.batch_size, overlap=overlap),
                      dataset.num_steps(batch=args.batch_size)],
            "eval":  [dataset.evaluation_set(batch_size=1, overlap=overlap),
                      dataset.num_steps(batch=1)]
            }[dataset_type]


def setup_callbacks(checkpoint_path, log_dir):
    """Returns callbacks for baseline training and optimization fine-tuning."""
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=1,  # save every epoch
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # update every epoch
            update_freq=100,  # update every 100 batch
        ),
    ]
    return callbacks


def get_lr_schedule(steps_per_epoch, learning_rate=1e-5, lr_schedule_config=[[1.0, 0.1, 0.01, 0.001]]):
    """Returns learn rate schedule for baseline training and optimization fine-tuning."""
    initial_learning_rate = learning_rate
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(p[1] * steps_per_epoch for p in lr_schedule_config),
        values=[initial_learning_rate] + list(p[0] * initial_learning_rate for p in lr_schedule_config))
    return lr_schedule


def prune_model(model):
    """Performs pruning, fine-tuning and returns stripped pruned model"""
    log("Pruning model to {} sparsity".format(args.sparsity))
    (training_data, training_num_steps) = get_data("train")
    (validation_data, validation_num_steps) = get_data("val")
    (evaluation_data, eval_num_steps) = get_data("eval")
    log_dir = "logs/pruned" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "checkpoints_pruned"
    export_checkpoint_path = "checkpoints_export"

    create_directories([log_dir, checkpoint_path, export_checkpoint_path])

    callbacks = setup_callbacks(os.path.join(checkpoint_path, "pruned-{epoch:04d}.h5"), log_dir)

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
            args.sparsity, begin_step=0, end_step=int(training_num_steps * 0.7), frequency=10
        )
    }

    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(steps_per_epoch=training_num_steps))
    pruned_model.compile(
        loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt)

    pruned_model.fit(
        training_data,
        epochs=args.finetuning_epochs,
        steps_per_epoch=training_num_steps,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_num_steps,
    )

    log("Evaluating {}".format(model.name))
    pruned_model.evaluate(x=evaluation_data, steps=eval_num_steps)

    stripped_model = tfmot.sparsity.keras.strip_pruning(model)

    stripped_model.save_weights(os.path.join(export_checkpoint_path, "pruned-{}.h5".format(str(args.finetuning_epochs))))

    return stripped_model


def prepare_model_for_inference(model):

    layer_input = tf.keras.layers.Input((296, 39), batch_size=1)
    static_shaped_model = tf.keras.models.Model(
        inputs=[layer_input], outputs=[model.call(layer_input)]
    )
    return static_shaped_model


def tflite_conversion(model, tflite_path, conversion_type="fp32"):
    """Performs tflite conversion (fp32, int8)."""
    # Prepare model for inference
    model = prepare_model_for_inference(model)

    create_directories([os.path.dirname(tflite_path)])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    def representative_dataset_gen(input_dim):
        calib_data = []
        for data in tqdm(training_data.take(1000), desc="model calibration"):
            input_data = data[0]
            for i in range(input_data.shape[1] // input_dim):
                input_chunks = [
                    input_data[:, i * input_dim: (i + 1) * input_dim, :, ]
                ]
            for chunk in input_chunks:
                calib_data.append([chunk])

        return lambda: [
            (yield data) for data in tqdm(calib_data, desc="model calibration")
        ]

    if conversion_type == "int8":
        log("Quantizing Model")
        (training_data, training_num_steps) = get_data("train", overlap=True)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset_gen(model.input_shape[1])

    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)


def evaluate_tflite(tflite_path):
    """Evaluates tflite (fp32, int8)."""
    results = []
    (evaluation_data, eval_num_steps) = get_data("eval")
    tflite_path = tflite_path

    log("Setting number of used threads to {}".format(multiprocessing.cpu_count()))
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path, num_threads=multiprocessing.cpu_count()
    )
    interpreter.allocate_tensors()
    input_chunk = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_chunk["shape"]
    log("eval_model() - input_shape: {}".format(input_shape))
    input_dtype = input_chunk["dtype"]
    output_dtype = output_details["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype != tf.float32:
        input_scale, input_zero_point = input_chunk["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    if output_dtype != tf.float32:
        output_scale, output_zero_point = output_details["quantization"]
    else:
        output_scale, output_zero_point = 1, 0

    log("Running {} iterations".format(eval_num_steps))
    for i_iter, (data, label) in enumerate(
            tqdm(evaluation_data, total=eval_num_steps)
    ):
        data = data / input_scale + input_zero_point
        # Round the data up if dtype is int8, uint8 or int16
        if input_dtype is not np.float32:
            data = np.round(data)

        while data.shape[1] < 296:
            data = np.append(data, data[:, -2:-1, :], axis=1)
        # Zero-pad any odd-length inputs
        if data.shape[1] % 2 == 1:
            log('Input length is odd, zero-padding to even (first layer has stride 2)')
            data = np.concatenate([data, np.zeros((1, 1, data.shape[2]), dtype=input_dtype)], axis=1)

        context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field on each side
        size = input_chunk['shape'][1]
        inner = size - 2 * context
        data_end = data.shape[1]

        # Initialize variables for the sliding window loop
        data_pos = 0
        outputs = []

        while data_pos < data_end:
            if data_pos == 0:
                # Align inputs from the first window to the start of the data and include the intial context in the output
                start = data_pos
                end = start + size
                y_start = 0
                y_end = y_start + (size - context) // 2
                data_pos = end - context
            elif data_pos + inner + context >= data_end:
                # Shift left to align final window to the end of the data and include the final context in the output
                shift = (data_pos + inner + context) - data_end
                start = data_pos - context - shift
                end = start + size
                assert start >= 0
                y_start = (shift + context) // 2  # Will be even because we assert it above
                y_end = size // 2
                data_pos = data_end
            else:
                # Capture only the inner region from mid-input inferences, excluding output from both context regions
                start = data_pos - context
                end = start + size
                y_start = context // 2
                y_end = y_start + inner // 2
                data_pos = end - context

            interpreter.set_tensor(
                input_chunk["index"], tf.cast(data[:, start:end, :], input_dtype))
            interpreter.invoke()
            cur_output_data = interpreter.get_tensor(output_details["index"])[:, :, y_start:y_end, :]
            cur_output_data = output_scale * (
                    cur_output_data.astype(np.float32) - output_zero_point
            )
            outputs.append(cur_output_data)

        complete = np.concatenate(outputs, axis=2)
        results.append(get_metrics("ler")(label, complete))

    log("Avg LER: {}".format(np.mean(results) * 100))


def main(args):
    """Main execution function"""
    # Model creation
    model = create_wav2letter(
        batch_size=args.batch_size, no_stride_count=args.no_stride_count
    )
    # load baseline wav2letter weights
    model.load_weights("weights/wav2letter.h5")

    # Prune model
    pruned_model = prune_model(model)

    # Pruned int8 wav2letter
    output_directory = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.as_posix()
    wav2letter_pruned_int8 = os.path.join(output_directory, "wav2letter_pruned_int8.tflite")
    tflite_conversion(pruned_model, wav2letter_pruned_int8, "int8")
    evaluate_tflite(wav2letter_pruned_int8)


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
    parser.add_argument(
        "--finetuning_epochs",
        dest="finetuning_epochs",
        type=int,
        required=False,
        default=1,
        help="Amount of epochs for baseline training",
    )
    parser.add_argument(
        "--no_stride_count",
        dest="no_stride_count",
        type=int,
        required=False,
        default=7,
        help="Number of Convolution2D layers without striding",
    )
    parser.add_argument(
        "--sparsity",
        dest="sparsity",
        type=float,
        required=False,
        default=0.5,
        help="Level of sparsity required",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    args = parser.parse_args()
    main(args)
