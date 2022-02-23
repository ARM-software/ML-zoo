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

"""Wav2letter training, optimisation and evaluation script"""
import argparse

import tensorflow as tf

from tinywav2letter import create_tinywav2letter, get_metrics
from load_mfccs import MFCC_Loader

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)


def log(std):
    """Log the given string to the standard output."""
    print("******* {}".format(std), flush=True)

def get_data(args, dataset_type, batch_size):
    """Returns particular training and validation dataset."""
    dataset = MFCC_Loader(args.full_data_dir, args.reduced_data_dir,args.fluent_speech_data_dir)

    return {"train_full_size": [dataset.full_training_set(batch_size=batch_size, num_samples = args.training_set_size).with_options(options), dataset.num_steps(batch=batch_size)],
            "train_reduced_size": [dataset.reduced_training_set(batch_size=batch_size, num_samples = args.training_set_size).with_options(options), dataset.num_steps(batch=batch_size)],
            "val_full_size": [dataset.full_validation_set(batch_size=batch_size).with_options(options), dataset.num_steps(batch=batch_size)],
            "val_reduced_size": [dataset.reduced_validation_set(batch_size=batch_size).with_options(options), dataset.num_steps(batch=batch_size)],
            "train_fluent_speech": [dataset.fluent_speech_train_set(batch_size=batch_size, num_samples = args.fluent_speech_set_size).with_options(options), dataset.num_steps(batch=batch_size)],
            "val_fluent_speech": [dataset.fluent_speech_validation_set(batch_size=batch_size).with_options(options), dataset.num_steps(batch=batch_size)],
            "test_fluent_speech": [dataset.fluent_speech_test_set(batch_size=batch_size).with_options(options),dataset.num_steps(batch=batch_size)],
            }[dataset_type]

def setup_callbacks(checkpoint_path, log_dir):
    """Returns callbacks for baseline training and optimization fine-tuning."""
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch',  # save every epoch
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # update every epoch
            update_freq=100,  # update every 100 batch

        ),
    ]
    return callbacks

def get_lr_schedule(steps_per_epoch, learning_rate=1e-4, lr_schedule_config=[[1.0, 0.1, 0.01, 0.001]]):
    """Returns learn rate schedule for baseline training and optimization fine-tuning."""
    initial_learning_rate = learning_rate
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(p[1] * steps_per_epoch for p in lr_schedule_config),
        values=[initial_learning_rate] + list(p[0] * initial_learning_rate for p in lr_schedule_config))
    return lr_schedule


def train_model(args):
    """Performs pruning, fine-tuning and returns stripped pruned model"""
    log("Commencing Model Training")
    
    # Get all of the required datasets
    (full_training_data, full_training_num_steps) = get_data(args, "train_full_size", args.batch_size)
    (reduced_training_data, reduced_training_num_steps) = get_data(args, "train_reduced_size", args.batch_size)
    (full_validation_data, full_validation_num_steps) = get_data(args, "val_full_size", args.batch_size)
    (reduced_validation_data, reduced_validation_num_steps) = get_data(args, "val_reduced_size", args.batch_size)
    (fluent_speech_training_data, fluent_speech_training_num_steps) = get_data(args, "train_fluent_speech", args.batch_size)
    (fluent_speech_validation_data, fluent_speech_validation_num_steps) = get_data(args, "val_fluent_speech", args.batch_size)
    (fluent_speech_test_data, fluent_speech_test_num_steps) = get_data(args, "test_fluent_speech",args.batch_size)

    # Set up checkpoint paths, directories for the log files and the callbacks
    baseline_checkpoint_path = f"checkpoints/baseline/checkpoint.ckpt"
    finetuning_checkpoint_path = f"checkpoints/finetuning/checkpoint.ckpt" 

    baseline_log_dir = f"logs/tiny_wav2letter_baseline"
    finetuning_log_dir = f"logs/tiny_wav2letter_finetuning"

    baseline_callbacks = setup_callbacks(baseline_checkpoint_path, baseline_log_dir)
    finetuning_callbacks = setup_callbacks(finetuning_checkpoint_path, finetuning_log_dir)

    # Initialise the Tiny Wav2Letter model
    with strategy.scope():
        model = create_tinywav2letter(batch_size = args.batch_size)


    # Perform the baseline training with the full size LibriSpeech dataset
    if args.with_baseline:

        with strategy.scope():
            opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-4, steps_per_epoch=full_training_num_steps))
            model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt)

        model.fit(
            full_training_data,
            epochs=args.baseline_epochs,
            verbose=1,
            callbacks=baseline_callbacks,
            validation_data=full_validation_data,
            initial_epoch = 0
        )

        log(f'Evaluating Tiny Wav2Letter post baseline training')
        model.evaluate(fluent_speech_test_data)

    # Perform finetuning training with the reduced size MiniLibriSpeech dataset
    if args.with_finetuning:

        with strategy.scope():
            opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-5, steps_per_epoch=reduced_training_num_steps))
            model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt)
          
        model.fit(
            reduced_training_data,
            epochs=args.finetuning_epochs + args.baseline_epochs,
            verbose=1,
            callbacks=finetuning_callbacks,
            validation_data=reduced_validation_data,
            initial_epoch = args.baseline_epochs
        )

        log(f'Evaluating Tiny Wav2Letter post finetuning')
        model.evaluate(x=fluent_speech_test_data)


    if args.with_fluent_speech:

        with strategy.scope():
            opt = tf.keras.optimizers.Adam(learning_rate=get_lr_schedule(learning_rate = 1e-5, steps_per_epoch=fluent_speech_training_num_steps))
            model.compile(loss=get_metrics("loss"), metrics=[get_metrics("ler")], optimizer=opt,run_eagerly=True)

        model.fit(
            fluent_speech_training_data,
            epochs=args.finetuning_epochs + args.baseline_epochs,
            verbose=1,
            callbacks=finetuning_callbacks,
            validation_data=fluent_speech_validation_data,
            initial_epoch = args.baseline_epochs
        )

        model.evaluate(x=fluent_speech_test_data)
    # Save the final trained model in TF SavedModel format
    model.save(f"saved_models/tiny_wav2letter")

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
        "--load_model",
        dest="load_model",
        required=False,
        action='store_true',
        help="Model number to load",
    )
    parser.add_argument(
        "--with_baseline",
        dest="with_baseline",
        required=False,
        action='store_true',
        help="Perform pre-training baseline using the full size dataset",
    )
    parser.add_argument(
        "--with_finetuning",
        dest="with_finetuning",
        required=False,
        action='store_true',
        help="Perform fine-tuning training using the reduced corpus dataset",
    )
    parser.add_argument(
        "--with_fluent_speech",
        dest="with_fluent_speech",
        required=False,
        action='store_true',
        help="Perform fluent_speech training using the fluent speech dataset",
    )
    parser.add_argument(
        "--baseline_epochs",
        dest="baseline_epochs",
        type=int,
        required=False,
        default=30,
        help="Number of epochs for baseline training",
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
        "--fluent_speech_epochs",
        dest="fluent_speech_epochs",
        type=int,
        required=False,
        default=30,
        help="Number of epochs for fluent speech training",
    )
    args = parser.parse_args()
    train_model(args)
