#  Copyright (c) 2021 Arm Limited. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Code for testing the quality of the trained RNNoise models."""
from pathlib import Path
import argparse
import logging

import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import pesq

from data import RNNoisePreProcess, load_wav, load_clean_noisy_wavs
from model import rnnoise_model


def tflite_forward(interpreter, input_data):
    """TFLite interpreter forward inference and return all outputs."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for index, data in enumerate(input_data):
        if input_details[index]["dtype"] == np.int8:
            input_scale, input_zero_point = input_details[index]["quantization"]
            data = data / input_scale + input_zero_point
            data = np.clip(data, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[index]['index'], data)

    interpreter.invoke()

    output = []
    for index in range(len(output_details)):
        out_data = interpreter.get_tensor(output_details[index]['index'])
        if output_details[index]["dtype"] == np.int8:
            output_scale, output_zero_point = output_details[index]["quantization"]
            out_data = output_scale * (out_data.astype(np.float32) - output_zero_point)

        output.append(out_data)

    return output


def denoise_wav_file(ckpt=None, tflite_model=None, noisy_wav=''):
    """Run a trained RNNoise model over a wav file and save the output."""
    if ckpt:
        model = rnnoise_model(timesteps=1, batch_size=1, is_training=False)
        model.load_weights(ckpt).expect_partial()
    elif tflite_model:
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
        interpreter.allocate_tensors()
        vad_gru_state = np.zeros((1, 24), dtype=np.float32)
        noise_gru_state = np.zeros((1, 48), dtype=np.float32)
        denoise_gru_state = np.zeros((1, 96), dtype=np.float32)
    else:
        RuntimeError("No ckpt file or tflite model supplied.")

    preprocess = RNNoisePreProcess(training=False)

    loaded_audio = load_wav(noisy_wav)

    final_denoised_audio = []
    num_samples = len(loaded_audio) // preprocess.FRAME_SIZE
    for i in range(num_samples):
        audio_window = loaded_audio[i * RNNoisePreProcess.FRAME_SIZE:
                                    i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]

        silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
        features = np.expand_dims(features, (0, 1)).astype(np.float32)

        if not silence:
            if ckpt:
                denoise_output = model(features)[0]
            elif tflite_model:
                input_data = [features, vad_gru_state, noise_gru_state, denoise_gru_state]
                model_output = tflite_forward(interpreter, input_data)
                denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model_output
                vad_gru_state = np.squeeze(vad_gru_state, axis=1)
                noise_gru_state = np.squeeze(noise_gru_state, axis=1)
                denoise_gru_state = np.squeeze(denoise_gru_state, axis=1)

            denoise_output = np.squeeze(np.array(denoise_output))
            denoised_audio_tmp = preprocess.post_process(silence, denoise_output, X, P, Ex, Ep, Exp)
            denoised_audio_tmp = np.rint(denoised_audio_tmp).astype(np.int16)
            final_denoised_audio.append(denoised_audio_tmp)
        else:
            final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))

    denoised_audio = np.concatenate(final_denoised_audio, axis=0)

    noisy_wav = Path(noisy_wav)
    sf.write(Path(noisy_wav.parent, f"{noisy_wav.stem}_denoised{noisy_wav.suffix}"), denoised_audio, 48000, 'PCM_16')
    logging.info(f"Denoising {noisy_wav} complete. Output saved to {noisy_wav.stem}_denoised{noisy_wav.suffix}")


def pesq_calculation(clean_wav, degraded_wav):
    """Calculate and returns pesq score between clean and degraded wav files."""
    # pesq library needs sample rate of 16000
    clean_audio = librosa.load(clean_wav, sr=16000)[0]
    degraded_audio = librosa.load(degraded_wav, sr=16000)[0]

    pesq_score = pesq.pesq(ref=clean_audio, deg=degraded_audio, fs=16000, mode='wb')

    return pesq_score


def denoise_and_calc_average_pesq_folder_wavs(clean_wavs, noisy_wavs, ckpt, tflite_model):
    """Run RNNoise over a folder of noisy wavs then calculate the average pesq after denoising.

    It is expected the clean speech folder and noisy speech folder have wav files with the same names.
    The wav file in the noisy speech folder will have the same speech but overlaid with some noise."""
    clean_speech_audio, noisy_speech_audio, _ = load_clean_noisy_wavs(clean_wavs, noisy_wavs, isolate_noise=False)

    sum_pesq = 0.0
    final_denoised_audio = []
    if ckpt:
        model = rnnoise_model(timesteps=1, batch_size=1, is_training=False)
        model.load_weights(ckpt).expect_partial()
    elif tflite_model:
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
        interpreter.allocate_tensors()
        vad_gru_state = np.zeros((1, 24), dtype=np.float32)
        noise_gru_state = np.zeros((1, 48), dtype=np.float32)
        denoise_gru_state = np.zeros((1, 96), dtype=np.float32)
    else:
        RuntimeError("No ckpt file or tflite model supplied.")

    for clean_speech, noisy_speech in zip(clean_speech_audio, noisy_speech_audio):
        preprocess = RNNoisePreProcess(training=False)

        num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
        for i in range(num_samples):
            audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]

            silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
            features = np.expand_dims(features, (0, 1)).astype(np.float32)

            if not silence:
                if ckpt:
                    denoise_output = model(features)[0]
                elif tflite_model:
                    input_data = [features, vad_gru_state, noise_gru_state, denoise_gru_state]
                    model_output = tflite_forward(interpreter, input_data)
                    denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model_output
                    vad_gru_state = np.squeeze(vad_gru_state, axis=1)
                    noise_gru_state = np.squeeze(noise_gru_state, axis=1)
                    denoise_gru_state = np.squeeze(denoise_gru_state, axis=1)

                denoise_output = np.squeeze(np.array(denoise_output))
                denoised_audio_tmp = preprocess.post_process(silence, denoise_output, X, P, Ex, Ep, Exp)
                denoised_audio_tmp = np.rint(denoised_audio_tmp)
                final_denoised_audio.append(denoised_audio_tmp)
            else:
                final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))

        denoised_audio = np.concatenate(final_denoised_audio, axis=0)

        resample_clean = librosa.resample(clean_speech, 48000, 16000)
        resample_denoised = librosa.resample(denoised_audio, 48000, 16000)

        sum_pesq += pesq.pesq(ref=resample_clean, deg=resample_denoised, fs=16000, mode='wb')

    return sum_pesq / len(clean_speech_audio)


def calc_average_pesq_folder_wavs(clean_speech_folder, noisy_speech_folder):
    """Caculate the average pesq between clean and noisy speech wav files.

    It is expected the clean speech folder and noisy speech folder have wav files with the same names.
    The wav file in the noisy speech folder will have the same speech but overlaid with some noise."""
    sum_pesq = 0.0
    num_tested_files = 0

    for clean_file in clean_speech_folder.rglob('*.wav'):
        try:
            sum_pesq += pesq_calculation(clean_file, noisy_speech_folder / clean_file.name)
            num_tested_files += 1
        except:
            logging.warning(f"Could not process {clean_file}, make sure it exists in both clean and noisy folders.")
            pass

    return sum_pesq / num_tested_files


def main():
    avg_pesq_noisy = calc_average_pesq_folder_wavs(Path(FLAGS.clean_wav_folder), Path(FLAGS.noisy_wav_folder))
    print(f"pesq score for audio before de-noising: {avg_pesq_noisy}")

    avg_pesq_denoised = denoise_and_calc_average_pesq_folder_wavs(Path(FLAGS.clean_wav_folder), Path(FLAGS.noisy_wav_folder),
                                                                  None, FLAGS.tflite_path)
    print(f"pesq score for audio after de-noising with {FLAGS.tflite_path} is: {avg_pesq_denoised}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clean_wav_folder',
        type=str,
        required=True,
        help='Path to the test set of clean wav files.'
    )
    parser.add_argument(
        '--noisy_wav_folder',
        type=str,
        required=True,
        help='Path to the test set of noisy wav files.'
    )
    parser.add_argument(
        '--tflite_path',
        type=str,
        required=True,
        help='Path to the TFLite model to use for removing noise.'
    )
    FLAGS, _ = parser.parse_known_args()
    main()
