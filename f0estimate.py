import os
import numpy as np
from scikits.audiolab import wavread
from scipy.signal import get_window

from utilities import nextpow2

class F0Estimate:

    def __init__(self, audio_path, **kwargs):
        if not os.path.exists(audio_path):
            raise ValueError('Invalid audio path')

        self.audio_path = audio_path

        # set maximum number of simultaneous notes
        if 'max_poly' in kwargs:
            self._max_poly = kwargs['max_poly']
        else:
            self._max_poly = 6

        # set minimum fundamental frequency to detect
        if 'min_f0' in kwargs:
            self._min_f0 = kwargs['min_f0']
        else:
            self._min_f0 = 40

        # set maximum fundamental frequency to detect
        if 'max_f0' in kwargs:
            self._max_f0 = kwargs['max_f0']
        else:
            self._max_f0 = 2100

        if 'method' in kwargs:
            self._method = kwargs['method']
            if self._method != 'iterative' and self._method != 'joint':
                raise ValueError('Unknown estimation method')
        else:
            self._method = 'iterative'

        # set analysis frame length
        if 'frame_len_sec' in kwargs:
            self._frame_len_sec = kwargs['frame_len_sec']
            if self._frame_len_sec != 0.046 and self._frame_len_sec != 0.093:
                raise ValueError('Analysis frame length must be 46ms or 93ms')
        else:
            self._frame_len_sec = 0.093

        if 'window_func' in kwargs:
            self._window_func = kwargs['window_func']
        else:
            self._window_func = 'hanning'

        # these parameter values are from the 2006 paper
        if self._frame_len_sec == 0.046:
            self._alpha = 27
            self._beta = 320
            self._d = 1.0
        else:
            self._alpha = 52
            self._beta = 320
            self._d = 0.89

    def gen_piano_roll(self, out_path):
        x, fs, _ = wavread(self.audio_path)

        # make x mono if stereo
        if x.ndim > 1:
            n_channels, _ = x.shape
            x = x.sum(axis=1)/n_channels

        X = self._stft(x, fs)

    def _stft(self, x, fs):
        '''
        Calculate short time fourier transform on signal that is
        hann windowed and zero-padded to twice its length.
        Hopsize = window length
        '''

        frame_len_samps = int(fs * self._frame_len_sec)
        win = get_window(self._window_func, frame_len_samps)

        # zero-pad to twice the length of the frame
        num_points = nextpow2(2*frame_len_samps)
        X = np.array([np.fft.fft(win*x[i:i+frame_len_samps], num_points)
                      for i in range(0, len(x)-frame_len_samps, frame_len_samps)])
        return X
