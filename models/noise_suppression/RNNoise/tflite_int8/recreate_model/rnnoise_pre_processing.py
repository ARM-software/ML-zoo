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
"""Python implementation of RNNoise feature calculation converting from C code https://github.com/xiph/rnnoise"""
import numpy as np


class RNNoisePreProcess:
    # Some constants for pre-processing specific to RNNoise.
    FRAME_SIZE_SHIFT = 2
    FRAME_SIZE = 480
    WINDOW_SIZE = 2 * FRAME_SIZE
    FREQ_SIZE = FRAME_SIZE + 1

    PITCH_MIN_PERIOD = 60
    PITCH_MAX_PERIOD = 768
    PITCH_FRAME_SIZE = 960
    PITCH_BUF_SIZE = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE

    NB_BANDS = 22
    CEPS_MEM = 8
    NB_DELTA_CEPS = 6

    NB_FEATURES = NB_BANDS + 3*NB_DELTA_CEPS + 2

    def __init__(self, training):
        self.training = training
        self.lowpass = self.FREQ_SIZE
        self.band_lp = self.NB_BANDS

        self.half_window = np.zeros(self.FRAME_SIZE)
        self.dct_table = np.zeros(self.NB_BANDS*self.NB_BANDS)

        # Denoise State params.
        self.analysis_mem = np.zeros(self.FRAME_SIZE)
        self.cepstral_mem = np.zeros((self.CEPS_MEM, self.NB_BANDS))
        self.memid = 0
        self.synthesis_mem = np.zeros(self.FRAME_SIZE)
        self.pitch_buf = np.zeros(self.PITCH_BUF_SIZE)
        self.pitch_enh_buf = np.zeros(self.PITCH_BUF_SIZE)
        self.last_gain = 0
        self.last_period = 0
        self.mem_hp_x = np.zeros(2)
        self.lastg = np.zeros(self.NB_BANDS)

        self._check_init()

        # 0 200 400 600 800 1k 1.2 1.4 1.6 2k 2.4 2.8 3.2 4k 4.8 5.6 6.8 8k 9.6 12k 15.6 20k
        self.eband5ms = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100)

    def post_process(self, silence, model_output, X, P, Ex, Ep, Exp):
        """Take model output and the outputs from pre-processing then produce de-noised audio."""
        g = model_output  # Gains

        if not silence:
            X = self._pitch_filter(X, P, Ex, Ep, Exp, g)
            for i in range(self.NB_BANDS):
                g[i] = max(g[i], 0.6*self.lastg[i])
                self.lastg[i] = g[i]

            gf = self._interp_band_gain(g)

            for i in range(self.FREQ_SIZE):
                X[i] *= gf[i]

        out_frame = self._frame_synythesis(X)
        return out_frame

    def process_frame(self, audio_window):
        """Process the input audio frame ready for inputting to RNNoise model."""

        # Apply a biquad filter over the input.
        a_hp = (-1.99599, 0.996)
        b_hp = (-2, 1)
        x, self.mem_hp_x = self._biquad(audio_window, b_hp=b_hp, a_hp=a_hp, mem_hp_x=self.mem_hp_x)

        # Calculate features and see if the audio window contains noise.
        silence, features, X, P, Ex, Ep, Exp = self._compute_frame_features(x)

        return silence, features, X, P, Ex, Ep, Exp

    def _check_init(self):
        """Additional initialization code.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L168"""
        for i in range(self.FRAME_SIZE):
            self.half_window[i] = np.sin(0.5*np.pi *
                                         np.sin(0.5*np.pi*(i+0.5)/self.FRAME_SIZE) *
                                         np.sin(0.5*np.pi*(i+0.5)/self.FRAME_SIZE))
        for i in range(self.NB_BANDS):
            for j in range(self.NB_BANDS):
                self.dct_table[i*self.NB_BANDS + j] = np.cos((i+0.5)*j*np.pi/self.NB_BANDS)
                if j == 0:
                    self.dct_table[i*self.NB_BANDS + j] *= np.sqrt(0.5)

    def _biquad(self, audio_window, b_hp, a_hp, mem_hp_x):
        """Apply a biquadratic filter to the audio window.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L410"""
        filter_applied = np.zeros_like(audio_window)

        for i in range(len(audio_window)):
            xi = audio_window[i]
            yi = audio_window[i] + mem_hp_x[0]
            mem_hp_x[0] = mem_hp_x[1] + (b_hp[0]*xi - a_hp[0]*yi)
            mem_hp_x[1] = (b_hp[1]*xi - a_hp[1]*yi)
            filter_applied[i] = yi

        return filter_applied, mem_hp_x

    def _compute_frame_features(self, audio_window):
        """Compute RNNoise features for the given audio window.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L310"""
        Ex, X, self.analysis_mem = self._frame_analysis(audio_window, self.analysis_mem)

        E = 0.0
        Ly = np.zeros(self.NB_BANDS)
        Exp = np.zeros(self.NB_BANDS)
        p = np.zeros(self.WINDOW_SIZE)

        pitch_buf = np.zeros(self.PITCH_BUF_SIZE >> 1)  # Half the pitch buffer size

        # Shift elements in the pitch buffer down and populate with the new audio window.
        self.pitch_buf[0:self.PITCH_BUF_SIZE-self.FRAME_SIZE] = self.pitch_buf[self.FRAME_SIZE:
                                                                               self.FRAME_SIZE+self.PITCH_BUF_SIZE-self.FRAME_SIZE]
        self.pitch_buf[self.PITCH_BUF_SIZE-self.FRAME_SIZE:self.PITCH_BUF_SIZE-self.FRAME_SIZE+self.FRAME_SIZE] = \
            audio_window[0:self.FRAME_SIZE]

        pitch_buf = self._pitch_downsample(pitch_buf, self.PITCH_BUF_SIZE)

        pitch_index = self._pitch_search(pitch_buf[self.PITCH_MAX_PERIOD >> 1:], pitch_buf, self.PITCH_FRAME_SIZE,
                                         self.PITCH_MAX_PERIOD-3*self.PITCH_MIN_PERIOD)

        pitch_index = self.PITCH_MAX_PERIOD-pitch_index

        gain, pitch_index = self._remove_doubling(pitch_buf, self.PITCH_MAX_PERIOD, self.PITCH_MIN_PERIOD,
                                                  self.PITCH_FRAME_SIZE, pitch_index, self.last_period, self.last_gain)

        self.last_period = pitch_index
        self.last_gain = gain

        for i in range(self.WINDOW_SIZE):
            p[i] = self.pitch_buf[self.PITCH_BUF_SIZE-self.WINDOW_SIZE-pitch_index+i]

        p = self._apply_window(p)
        P = self._forward_transform(p)
        Ep = self._compute_band_energy(P)
        Exp = self._compute_band_corr(Exp, X, P)

        for i in range(self.NB_BANDS):
            Exp[i] = Exp[i] / np.sqrt(0.001 + Ex[i]*Ep[i])

        tmp = np.zeros(self.NB_BANDS)
        tmp = self._dct(tmp, Exp)

        features = np.zeros(self.NB_FEATURES)
        for i in range(self.NB_DELTA_CEPS):
            features[self.NB_BANDS + 2*self.NB_DELTA_CEPS + i] = tmp[i]

        features[self.NB_BANDS + 2*self.NB_DELTA_CEPS] -= 1.3
        features[self.NB_BANDS + 2*self.NB_DELTA_CEPS + 1] -= 0.9
        features[self.NB_BANDS + 3*self.NB_DELTA_CEPS] = 0.01 * (pitch_index - 300)

        logMax = -2
        follow = -2
        for i in range(self.NB_BANDS):
            Ly[i] = np.log10(1e-2 + Ex[i])
            Ly[i] = max(logMax-7, max(follow-1.5, Ly[i]))
            logMax = max(logMax, Ly[i])
            follow = max(follow-1.5, Ly[i])
            E += Ex[i]

        if (not self.training) and E < 0.04:
            # If there's no audio avoid messing up the state.
            features = np.zeros_like(features)
            return True, features

        features = self._dct(features, Ly)
        features[0] -= 12
        features[1] -= 4

        ceps_1 = self.cepstral_mem[self.CEPS_MEM+self.memid-1, :] if self.memid < 1 else self.cepstral_mem[self.memid-1, :]
        ceps_2 = self.cepstral_mem[self.CEPS_MEM+self.memid-2, :] if self.memid < 2 else self.cepstral_mem[self.memid-2, :]

        # Ceps_0
        for i in range(self.NB_BANDS):
            self.cepstral_mem[self.memid, i] = features[i]

        for i in range(self.NB_DELTA_CEPS):
            features[i] = self.cepstral_mem[self.memid, i] + ceps_1[i] + ceps_2[i]
            features[self.NB_BANDS + i] = self.cepstral_mem[self.memid, i] - ceps_2[i]
            features[self.NB_BANDS+self.NB_DELTA_CEPS+i] = self.cepstral_mem[self.memid, i] - 2*ceps_1[i] + ceps_2[i]

        # Spectral variability features.
        self.memid += 1
        if self.memid == self.CEPS_MEM:
            self.memid = 0
        spec_variability = 0

        for i in range(self.CEPS_MEM):
            mindist = 1e15
            for j in range(self.CEPS_MEM):
                dist = 0
                for k in range(self.NB_BANDS):
                    tmp = self.cepstral_mem[i, k] - self.cepstral_mem[j, k]
                    dist += tmp*tmp

                if j != i:
                    mindist = min(mindist, dist)
            spec_variability += mindist

        features[self.NB_BANDS + 3*self.NB_DELTA_CEPS + 1] = spec_variability / self.CEPS_MEM - 2.1

        return (self.training and E < 0.1), features, X, P, Ex, Ep, Exp

    def _frame_analysis(self, audio_window, analysis_mem):
        """Return band energy and calculated FFT.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L295"""
        x = np.zeros(self.WINDOW_SIZE)

        # Move old audio down and populate end with latest audio window.
        x[0:self.FRAME_SIZE] = analysis_mem[0:self.FRAME_SIZE]
        for i in range(self.FRAME_SIZE):
            x[self.FRAME_SIZE + i] = audio_window[i]
        analysis_mem[0:self.FRAME_SIZE] = audio_window[0:self.FRAME_SIZE]

        x = self._apply_window(x)
        # Calculate FFT
        X = self._forward_transform(x)

        if self.training:
            for i in range(self.lowpass, self.FREQ_SIZE):
                X[i] = 0 + 1j*0

        Ex = self._compute_band_energy(X)

        return Ex, X, analysis_mem

    def _apply_window(self, x):
        """Multiply input by sinosoidal function.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L247"""
        for i in range(self.FRAME_SIZE):
            x[i] *= self.half_window[i]
            x[self.WINDOW_SIZE - 1 - i] *= self.half_window[i]

        return x

    def _forward_transform(self, x):
        """Calculate FFT transform.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L212"""
        num_fft = 2*self.FRAME_SIZE

        fft_out = np.fft.fft(x, num_fft) / num_fft

        # Only want to take FREQ_SIZE elements of the FFT.
        return fft_out[0:self.FREQ_SIZE]

    def _compute_band_energy(self, fft_X):
        """Calculate energy in different frequency bands.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L100"""
        sum_arr = np.zeros(self.NB_BANDS)

        for i in range(self.NB_BANDS-1):
            band_size = (self.eband5ms[i+1]-self.eband5ms[i]) << self.FRAME_SIZE_SHIFT
            for j in range(band_size):
                frac = float(j) / band_size
                tmp = np.square(np.real(fft_X[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j]))
                tmp += np.square(np.imag(fft_X[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j]))
                sum_arr[i] += (1-frac) * tmp
                sum_arr[i+1] += frac * tmp

        sum_arr[0] *= 2
        sum_arr[self.NB_BANDS-1] *= 2

        return sum_arr

    def _pitch_downsample(self, local_pitch_buf, buf_len):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L148"""

        for i in range(1, buf_len >> 1):
            local_pitch_buf[i] = 0.5*(0.5*(self.pitch_buf[2*i - 1] + self.pitch_buf[2*i + 1]) + self.pitch_buf[2*i])
        local_pitch_buf[0] = 0.5*(0.5*(self.pitch_buf[1]) + self.pitch_buf[0])

        ac = np.zeros(5)
        num_lags = 4
        shift, ac = self._autocorr(local_pitch_buf, ac, num_lags, buf_len >> 1)

        # Noise floor -40db.
        ac[0] *= 1.0001

        # Lag windowing.
        for i in range(1, num_lags+1):
            ac[i] -= ac[i] * (0.008*i) * (0.008*i)

        lpc = np.zeros(num_lags)
        lpc = self._lpc(lpc, ac, num_lags)

        tmp = 1.0
        for i in range(num_lags):
            tmp = 0.9 * tmp
            lpc[i] = lpc[i] * tmp

        lpc2 = np.zeros(num_lags + 1)
        c1 = 0.8

        # Add a zero.
        lpc2[0] = lpc[0] + 0.8
        lpc2[1] = lpc[1] + (c1 * lpc[0])
        lpc2[2] = lpc[2] + (c1 * lpc[1])
        lpc2[3] = lpc[3] + (c1 * lpc[2])
        lpc2[4] = (c1 * lpc[3])

        mem = np.zeros(5)
        x, mem = self._fir5(local_pitch_buf, lpc2, buf_len >> 1, mem)

        return x

    def _autocorr(self, x, ac, lag, n):
        """Calculate auto correlations.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/celt_lpc.c#L198"""
        fastN = n-lag
        shift = 0

        # Auto-correlation (can be done with a numpy function?)
        ac = self._pitch_xcorr(x, x, ac, fastN, lag+1)

        # Modify auto-correlation by summing with auto-correlation for different lags.
        for k in range(lag+1):
            d = 0
            for i in range(k+fastN, n):
                d += x[i] * x[i-k]
            ac[k] += d

        return shift, ac

    def _pitch_xcorr(self, _x, _y, ac, _len, max_pitch):
        """Naive cross correlation calculation.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L218"""
        for i in range(max_pitch):
            sum_ = 0
            for j in range(_len):
                sum_ += _x[j] * _y[i + j]
            ac[i] = sum_

        return ac

    def _lpc(self, _lpc, ac, p):
        """Calculate linear predictor coefficients.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/celt_lpc.c#L37"""
        lpc = np.zeros_like(_lpc)
        error = ac[0]

        if ac[0] != 0:
            for i in range(p):
                # Sum up this iteration's reflection coefficient
                rr = 0
                for j in range(i):
                    rr += lpc[j] * ac[i - j]
                rr += ac[i + 1]
                r = -rr / error

                # Update LP coeffieicents and total error
                lpc[i] = r
                for j in range((i + 1) >> 1):
                    tmp1 = lpc[j]
                    tmp2 = lpc[i - 1 - j]
                    lpc[j] = tmp1 + (r * tmp2)
                    lpc[i - 1 - j] = tmp2 + (r * tmp1)

                error = error - (r * r * error)
                # Bail out once we get 30dB gain
                if error < 0.001 * ac[0]:
                    break
        return lpc

    def _fir5(self, x, num, N, mem):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L106"""
        num0 = num[0]
        num1 = num[1]
        num2 = num[2]
        num3 = num[3]
        num4 = num[4]
        mem0 = mem[0]
        mem1 = mem[1]
        mem2 = mem[2]
        mem3 = mem[3]
        mem4 = mem[4]

        for i in range(N):
            sum_ = x[i] + num0*mem0 + num1*mem1 + num2*mem2 + num3*mem3 + num4*mem4
            mem4 = mem3
            mem3 = mem2
            mem2 = mem1
            mem1 = mem0
            mem0 = x[i]
            x[i] = sum_

        mem[0] = mem0
        mem[1] = mem1
        mem[2] = mem2
        mem[3] = mem3
        mem[4] = mem4

        return x, mem

    def _pitch_search(self, x_lp, y, len_, max_pitch):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L283"""
        lag = len_ + max_pitch
        x_lp4 = np.zeros(len_ >> 2)
        y_lp4 = np.zeros(lag >> 2)
        xcorr = np.zeros(max_pitch >> 1)

        # Downsample by 2 again.
        for j in range(len_ >> 2):
            x_lp4[j] = x_lp[2*j]
        for j in range(lag >> 2):
            y_lp4[j] = y[2*j]

        xcorr = self._pitch_xcorr(x_lp4, y_lp4, xcorr, len_ >> 2, max_pitch >> 2)

        # Coarse search with 4x decimation.
        best_pitch = self._find_best_pitch(xcorr, y_lp4, len_ >> 2, max_pitch >> 2)

        # Finer search with 2x decimation.
        for i in range(max_pitch >> 1):
            xcorr[i] = 0
            if abs(i-2*best_pitch[0]) > 2 and abs(i-2*best_pitch[1]) > 2:
                continue
            sum_ = np.dot(x_lp[0:len_ >> 1], y[i:(len_ >> 1)+i])
            xcorr[i] = max(-1, sum_)

        best_pitch = self._find_best_pitch(xcorr, y, len_ >> 1, max_pitch >> 1)

        # Refine by pseudo-interpolation.
        if 0 < best_pitch[0] < ((max_pitch >> 1) - 1):
            a = xcorr[best_pitch[0] - 1]
            b = xcorr[best_pitch[0]]
            c = xcorr[best_pitch[0] + 1]
            if (c-a) > 0.7*(b-a):
                offset = 1
            elif (a-c) > 0.7*(b-c):
                offset = -1
            else:
                offset = 0
        else:
            offset = 0

        pitch = 2*best_pitch[0] - offset

        return pitch

    def _find_best_pitch(self, xcorr, y, len_, max_pitch):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L46"""
        Syy = 1
        best_num = [-1, -1]
        best_den = [0, 0]
        best_pitch = [0, 1]

        for j in range(len_):
            Syy += (y[j] * y[j])

        for i in range(max_pitch):
            if xcorr[i] > 0:
                xcorr16 = xcorr[i] * 1e-12  # Avoid problems when squaring.

                num = xcorr16 * xcorr16
                if num*best_den[1] > best_num[1]*Syy:
                    if num*best_den[0] > best_num[0]*Syy:
                        best_num[1] = best_num[0]
                        best_den[1] = best_den[0]
                        best_pitch[1] = best_pitch[0]
                        best_num[0] = num
                        best_den[0] = Syy
                        best_pitch[0] = i
                    else:
                        best_num[1] = num
                        best_den[1] = Syy
                        best_pitch[1] = i
            Syy += (y[i+len_]*y[i+len_]) - (y[i]*y[i])
            Syy = max(1, Syy)

        return best_pitch

    def _remove_doubling(self, x, maxperiod, minperiod, N, T0_, prev_period, prev_gain):
        """Remove pitch period doubling errors.
        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L423"""
        second_check = (0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2)
        minperiod0 = minperiod
        maxperiod //= 2
        minperiod //= 2
        T0_ //= 2
        prev_period //= 2
        N //= 2
        x_start = maxperiod
        if T0_ >= maxperiod:
            T0_ = maxperiod - 1

        T = T0 = T0_
        xx = np.dot(x[x_start:x_start+N], x[x_start:x_start+N])
        xy = np.dot(x[x_start:x_start+N], x[x_start-T0:x_start-T0+N])
        yy_lookup = np.zeros(maxperiod+1)
        yy_lookup[0] = xx
        yy = xx

        for i in range(1, maxperiod+1):
            yy = yy + (x[x_start-i] * x[x_start-i]) - (x[x_start+N-i] * x[x_start+N-i])
            yy_lookup[i] = max(0, yy)

        yy = yy_lookup[T0]
        best_xy = xy
        best_yy = yy

        g = g0 = self._compute_pitch_gain(xy, xx, yy)

        # Look for any pitch at T/k.
        for k in range(2, 16):
            T1 = int((2*T0+k)/(2*k))
            if T1 < minperiod:
                break
            # Look for another strong correlation at T1b.
            if k == 2:
                if (T1+T0) > maxperiod:
                    T1b = T0
                else:
                    T1b = T0 + T1
            else:
                T1b = (2*second_check[k]*T0 + k) // (2*k)

            xy = np.dot(x[x_start:x_start + N], x[x_start - T1:x_start - T1 + N])
            xy2 = np.dot(x[x_start:x_start + N], x[x_start - T1b:x_start - T1b + N])
            xy = 0.5 * (xy + xy2)
            yy = 0.5 * (yy_lookup[T1] + yy_lookup[T1b])
            g1 = self._compute_pitch_gain(xy, xx, yy)

            if abs(T1-prev_period) <= 1:
                cont = prev_gain
            elif abs(T1-prev_period) <= 2 and 5*k*k < T0:
                cont = 0.5*prev_gain
            else:
                cont = 0
            thresh = max(0.3, 0.7*g0-cont)

            # Bias against very high pitch (very short period) to avoid false-positives due to short-term correlation
            if T1 < 3*minperiod:
                thresh = max(0.4, 0.85*g0-cont)
            elif T1 < 2*minperiod:
                thresh = max(0.5, 0.9*g0-cont)
            if g1 > thresh:
                best_xy = xy
                best_yy = yy
                T = T1
                g = g1

        best_xy = max(0, best_xy)
        if best_yy <= best_xy:
            pg = 1.0
        else:
            pg = best_xy/(best_yy+1)

        xcorr = np.zeros(3)
        for k in range(3):
            xcorr[k] = np.dot(x[x_start:x_start+N], x[x_start-(T+k-1):x_start-(T+k-1)+N])
        if (xcorr[2]-xcorr[0]) > 0.7*(xcorr[1]-xcorr[0]):
            offset = 1
        elif (xcorr[0]-xcorr[2]) > 0.7*(xcorr[1]-xcorr[2]):
            offset = -1
        else:
            offset = 0

        if pg > g:
            pg = g

        T0_ = 2*T + offset

        if T0_ < minperiod0:
            T0_ = minperiod0

        return pg, T0_

    def _compute_pitch_gain(self, xy, xx, yy):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/pitch.c#L416"""
        return xy/np.sqrt(1+xx*yy)

    def _compute_band_corr(self, bandE, X, P):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L125"""
        sum_ = np.zeros(self.NB_BANDS)
        for i in range(self.NB_BANDS-1):
            band_size = (self.eband5ms[i+1] - self.eband5ms[i]) << self.FRAME_SIZE_SHIFT

            for j in range(band_size):
                frac = float(j)/band_size
                tmp = np.real(X[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j]) * \
                    np.real(P[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j])
                tmp += np.imag(X[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j]) * \
                    np.imag(P[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j])
                sum_[i] += (1-frac)*tmp
                sum_[i+1] += frac*tmp
        sum_[0] *= 2
        sum_[self.NB_BANDS-1] *= 2

        for i in range(self.NB_BANDS):
            bandE[i] = sum_[i]

        return bandE

    def _dct(self, out_, in_):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L184"""
        for i in range(self.NB_BANDS):
            sum_ = 0
            for j in range(self.NB_BANDS):
                sum_ += in_[j] * self.dct_table[j*self.NB_BANDS + i]

            out_[i] = sum_ * np.sqrt(2.0/22)

        return out_

    def _pitch_filter(self, X, P, Ex, Ep, Exp, g):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L422"""
        r = np.zeros(self.NB_BANDS)

        for i in range(self.NB_BANDS):
            if Exp[i] > g[i]:
                r[i] = 1
            else:
                r[i] = np.square(Exp[i])*(1-np.square(g[i]))/(0.001 + np.square(g[i])*(1-np.square(Exp[i])))

            r[i] = np.sqrt(min(1, max(0, r[i])))
            r[i] *= np.sqrt(Ex[i] / (1e-8+Ep[i]))

        rf = self._interp_band_gain(r)
        for i in range(self.FREQ_SIZE):
            X[i] += rf[i]*P[i]

        newE = self._compute_band_energy(X)
        norm = np.zeros(self.NB_BANDS)

        for i in range(self.NB_BANDS):
            norm[i] = np.sqrt(Ex[i]/(1e-8+newE[i]))

        normf = self._interp_band_gain(norm)
        for i in range(self.FREQ_SIZE):
            X[i] *= normf[i]

        return X

    def _interp_band_gain(self, bandE):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L150"""
        g = np.zeros(self.FREQ_SIZE)

        for i in range(self.NB_BANDS-1):
            band_size = (self.eband5ms[i+1]-self.eband5ms[i]) << self.FRAME_SIZE_SHIFT
            for j in range(band_size):
                frac = float(j)/band_size
                g[(self.eband5ms[i] << self.FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1]

        return g

    def _inverse_transform(self, fft_x_in):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L227"""
        x = np.zeros(self.WINDOW_SIZE, dtype=np.complex128)
        out = np.zeros(self.WINDOW_SIZE)

        for i in range(self.FREQ_SIZE):
            x[i] = fft_x_in[i]

        for i in range(self.FREQ_SIZE, self.WINDOW_SIZE):
            x[i] = np.conj(x[self.WINDOW_SIZE - i])

        num_fft = 2 * self.FRAME_SIZE
        fft_out = np.fft.fft(x, num_fft) / num_fft

        # Output in reverse order for IFFT.
        out[0] = self.WINDOW_SIZE * fft_out[0].real
        for i in range(1, self.WINDOW_SIZE):
            out[i] = self.WINDOW_SIZE * fft_out[self.WINDOW_SIZE - i].real

        return out

    def _frame_synythesis(self, fft_y):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L401"""
        x = self._inverse_transform(fft_y)
        x = self._apply_window(x)

        out = np.zeros(self.FRAME_SIZE)
        for i in range(self.FRAME_SIZE):
            out[i] = x[i] + self.synthesis_mem[i]

        self.synthesis_mem[0:self.FRAME_SIZE] = x[self.FRAME_SIZE:]

        return out

    def _uni_rand(self):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L498"""
        return np.random.random() - 0.5

    def _rand_resp(self, a, b):
        """Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L502"""
        a[0] = 0.75 * self._uni_rand()
        a[1] = 0.75 * self._uni_rand()
        b[0] = 0.75 * self._uni_rand()
        b[1] = 0.75 * self._uni_rand()

        return a, b

    def get_training_features(self, clean_audio_window, noise_audio_window, max_count):
        """For some clean audio generate the features needed to train (input to model, expected outputs).

        Max_count number of training features will be generated
        Currently all audio should be passed in as one big audio array.

        Ref: https://github.com/xiph/rnnoise/blob/1cbdbcf1283499bbb2230a6b0f126eb9b236defd/src/denoise.c#L509"""
        input_features = []
        output_labels = []
        vad_labels = []
        a_hp = (-1.99599, 0.996)
        b_hp = (-2, 1)
        clean_analysis_mem = np.zeros(self.FRAME_SIZE)
        a_noise = np.zeros(2)
        b_noise = np.zeros(2)
        a_sig = np.zeros(2)
        b_sig = np.zeros(2)
        mem_hp_x = np.zeros(2)
        mem_hp_n = np.zeros(2)
        mem_resp_x = np.zeros(2)
        mem_resp_n = np.zeros(2)

        noise_gain = 1
        speech_gain = 1
        vad_cnt = 0
        gain_change_count = 0
        count = 0

        while True:
            if count == max_count:
                break
            if count % 1000 == 0:
                print(f"Generating training data, count: {count}")
            E = 0
            g = np.zeros(self.NB_BANDS)

            # Random augmentation.
            gain_change_count += 1
            if gain_change_count > 2821:
                speech_gain = 10**((-40+((np.random.random()*32767) % 60)) / 20)
                noise_gain = 10**((-30+((np.random.random()*32767) % 50)) / 20)
                if np.random.random()*32767 % 10 == 0:
                    noise_gain *= speech_gain
                if np.random.random()*32767 % 10 == 0:
                    gain_change_count = 0
                a_noise, b_noise = self._rand_resp(a_noise, b_noise)
                a_sig, b_sig = self._rand_resp(a_sig, b_sig)
                lowpass = self.FREQ_SIZE * (3000.0/24000) * 50**(np.random.random())
                for i in range(self.NB_BANDS):
                    if (self.eband5ms[i] << self.FRAME_SIZE_SHIFT) > lowpass:
                        self.band_lp = i
                        break

            tmp = clean_audio_window[count*self.FRAME_SIZE:count*self.FRAME_SIZE + self.FRAME_SIZE]
            x = speech_gain * tmp

            E += np.sum(np.square(tmp))

            tmp = noise_audio_window[count*self.FRAME_SIZE:count*self.FRAME_SIZE + self.FRAME_SIZE]
            n = noise_gain * tmp

            x, mem_hp_x = self._biquad(x, b_hp=b_hp, a_hp=a_hp, mem_hp_x=mem_hp_x)
            x, mem_resp_x = self._biquad(x, b_hp=b_sig, a_hp=a_sig, mem_hp_x=mem_resp_x)
            n, mem_hp_n = self._biquad(n, b_hp=b_hp, a_hp=a_hp, mem_hp_x=mem_hp_n)
            n, mem_resp_n = self._biquad(n, b_hp=b_noise, a_hp=a_noise, mem_hp_x=mem_resp_n)

            xn = x + n

            if E > 1e9:
                vad_cnt = 0
            elif E > 1e8:
                vad_cnt -= 5
            elif E > 1e7:
                vad_cnt += 1
            else:
                vad_cnt += 2

            vad_cnt = min(max(0, vad_cnt), 15)

            if vad_cnt >= 10:
                vad = 0
            elif vad_cnt > 0:
                vad = 0.5
            else:
                vad = 1.0

            Ey, Y, clean_analysis_mem = self._frame_analysis(x, clean_analysis_mem)

            silence, features, X, P, Ex, Ep, Exp = self._compute_frame_features(xn)

            for i in range(self.NB_BANDS):
                g[i] = np.sqrt((Ey[i]+1e-3) / (Ex[i]+1e-3))
                if g[i] > 1:
                    g[i] = 1
                if silence or i > self.band_lp:
                    g[i] = -1
                if Ey[i] < 5e-2 and Ex[i] < 5e-2:
                    g[i] = -1
                if vad == 0 and noise_gain == 0:
                    g[i] = -1

            count += 1
            input_features.append(features)
            output_labels.append(g)
            vad_labels.append(vad)

        return input_features, output_labels, vad_labels
