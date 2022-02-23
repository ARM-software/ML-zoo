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
import multiprocessing
import os
import pathlib

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm
import numpy as np

from tinywav2letter import get_metrics
from train_model import log, get_lr_schedule, get_data

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

def load_model():
    """
        Returns the model saved at the end of training
    """

    # load the saved model
    with strategy.scope():
        model = tf.keras.models.load_model(f'saved_models/tiny_wav2letter', 
            custom_objects={'ctc_loss': get_metrics("loss"), 'ctc_ler':get_metrics("ler")}
            )
    
    return model

def evaluate_model(args, model):
    """
    Evaluates an unquantised model

    Args:
    args: The command line arguments
    model: The model to evaluate
    """
    
    # Get the data to evaluate the model on
    (fluent_speech_test_data, fluent_speech_test_num_steps) = get_data(args, "test_fluent_speech", args.batch_size)

    # Compile and evaluate the model - LER
    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-6, steps_per_epoch=fluent_speech_test_num_steps))
        model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt, run_eagerly=True)
        log(f'Evaluating TinyWav2Letter - LER')
        model.evaluate(fluent_speech_test_data)

    # Get the data to evaluate the model on
    (fluent_speech_test_data, fluent_speech_test_num_steps) = get_data(args, "test_fluent_speech", batch_size=1) # #based on batch=1

    # Compile and evaluate the model - WER
    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-6, steps_per_epoch=fluent_speech_test_num_steps))
        model.compile(loss=get_metrics("loss"),  metrics=[get_metrics("wer")], optimizer=opt,run_eagerly=True)
        log(f'Evaluating TinyWav2Letter - WER')
        model.evaluate(fluent_speech_test_data)

def prune_model(args, model):
    """Performs pruning, fine-tuning and returns stripped pruned model"""

    # Get all the training and validation data
    (full_training_data, full_training_num_steps) = get_data(args, "train_full_size", args.batch_size)
    (full_validation_data, full_validation_num_steps) = get_data(args, "val_full_size", args.batch_size)
    (fluent_speech_training_data, fluent_speech_training_num_steps) = get_data(args, "train_fluent_speech",args.batch_size)
    (fluent_speech_validation_data, fluent_speech_validation_num_steps) = get_data(args, "val_fluent_speech",args.batch_size)
    (fluent_speech_test_data, fluent_speech_test_num_steps) = get_data(args, "test_fluent_speech", args.batch_size)

    log("Pruning model to {} sparsity".format(args.sparsity))
    
    log_dir = f"logs/pruned_model"

    # Set up the callbacks
    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
        ]
        
    # Perform the pruning - full_training_data
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
            args.sparsity, begin_step=0, end_step=int(full_training_num_steps * 0.7), frequency=10
        )
    }

    
    with strategy.scope():
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-4, steps_per_epoch=full_training_num_steps))
        pruned_model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt)

    pruned_model.fit(
        full_training_data,
        epochs=5,
        verbose=1,
        callbacks=pruning_callbacks,
        validation_data=full_validation_data,
    )

    # Perform the pruning - fluent_speech_training_data
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
            args.sparsity, begin_step=0, end_step=int(fluent_speech_validation_num_steps * 0.7), frequency=10
        )
    }

    
    with strategy.scope():
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-4, steps_per_epoch=fluent_speech_validation_num_steps))
        pruned_model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt)

    pruned_model.fit(
        fluent_speech_training_data,
        epochs=5,
        verbose=1,
        callbacks=pruning_callbacks,
        validation_data=fluent_speech_validation_data,
    )

    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    return stripped_model

def prepare_model_for_inference(model, input_window_length =  296):
    "Takes the a model and returns a model with fixed input size"
    MFCC_coeffs = 39

    # Define the input
    layer_input = tf.keras.layers.Input((input_window_length, MFCC_coeffs), batch_size=1)
    static_shaped_model = tf.keras.models.Model(
        inputs=[layer_input], outputs=[model.call(layer_input)]
    )
    return static_shaped_model

def tflite_conversion(model, tflite_path, conversion_type="fp32"):
    """Performs tflite conversion (fp32, int8)."""
    # Prepare model for inference
    model = prepare_model_for_inference(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # define a dataset to calibrate the conversion to INT8
    def representative_dataset_gen(input_dim):
        calib_data = []
        for data in tqdm(fluent_speech_test_data.take(1000), desc="model calibration"):
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
        (fluent_speech_test_data, fluent_speech_test_num_steps) = get_data(args, "test_fluent_speech", args.batch_size)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset_gen(model.input_shape[1])

    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)

def evaluate_tflite(tflite_path, input_window_length =  296):
    """Evaluates tflite (fp32, int8)."""
    results_ler = []
    results_wer = []
    (fluent_speech_test_data, fluent_speech_test_num_steps) = get_data(args, "test_fluent_speech", batch_size=1)
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

    log("Running {} iterations".format(fluent_speech_test_num_steps))
    for i_iter, (data, label) in enumerate(
            tqdm(fluent_speech_test_data, total=fluent_speech_test_num_steps)
    ):
        data = data / input_scale + input_zero_point
        # Round the data up if dtype is int8, uint8 or int16
        if input_dtype is not np.float32:
            data = np.round(data)

        while data.shape[1] < input_window_length:
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
        results_ler.append(get_metrics("ler")(label, complete))
        results_wer.append(get_metrics("wer")(label, complete))
    
    log("ler: {}".format(np.mean(results_ler)))
    log("wer: {}".format(np.mean(results_wer))) #based on batch=1

def main(args):

    model = load_model()
    evaluate_model(args, model)

    if args.prune:
        model = prune_model(args, model)
        model.save(f"saved_models/pruned_tiny_wav2letter")
        evaluate_model(args, model)

    output_directory = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    output_directory = os.path.join(output_directory, "tiny_wav2letter/tflite_models")
    wav2letter_tflite_path = os.path.join(output_directory, args.prune * "pruned_" + f"tiny_wav2letter_int8.tflite")

    if not os.path.exists(os.path.dirname(wav2letter_tflite_path)):
        try:
            os.makedirs(os.path.dirname(wav2letter_tflite_path))
        except OSError as exc:
                raise

    # Convert the saved model to TFLite and then evaluate
    tflite_conversion(model, wav2letter_tflite_path, "int8")
    evaluate_tflite(wav2letter_tflite_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--training_set_size",
        dest="training_set_size",
        type=int,
        required=False,
        default = 132553,
        help="The number of samples in the training set"
    )
    parser.add_argument(
        "--fluent_speech_training_set_size",
        dest="fluent_speech_set_size",
        type=int,
        required=False,
        default = 23132,
        help="The number of samples in the fluent speech training dataset"
    )
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
        default=10,
        help="Number of epochs for fine-tuning",
    )
    parser.add_argument(
        "--full_data_dir",
        dest="full_data_dir",
        type=str,
        required=False,
        default='librispeech_full_size',
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--reduced_data_dir",
        dest="reduced_data_dir",
        type=str,
        required=False,
        default='librispeech_reduced_size',
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--fluent_speech_data_dir",
        dest="fluent_speech_data_dir",
        type=str,
        required=False,
        default='fluent_speech_commands_dataset',
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--prune",
        dest="prune",
        required=False,
        action='store_true',
        help="Prune model true or false",
    )
    parser.add_argument(
        "--sparsity",
        dest="sparsity",
        type=float,
        required=False,
        default=0.5,
        help="Level of sparsity required",
    )

    args = parser.parse_args()
    main(args)
