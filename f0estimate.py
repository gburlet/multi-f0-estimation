import os
import numpy as np
from scikits.audiolab import wavread

class F0Estimate:

    def __init__(self, audio_path, **kwargs):
        if os.path.exists(audio_path):
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

        if 'method' in kwarg:
            self._method = kwargs['method']
            if self._method != 'iterative' and self._method != 'joint'
                raise ValueError('Unknown estimation method')
        else:
            self._method = 'iterative'

        # set analysis frame length
        if 'frame_len' in kwargs:
            self._frame_len = kwargs['frame_len']
            if self._frame_len != 0.046 and self._frame_len != 0.093:
                raise ValueError('Analysis frame length must be 46ms or 93ms')
        else:
            self._frame_len = 0.093

        if 'window_func' in kwargs:
            self._window_func = kwargs['window_func']
        else:
            self._window_func = 'hann'

        # these parameter values are from the 2006 paper
        if self._frame_len == 0.046:
            self._alpha = 27
            self._beta = 320
            self._d = 1.0
        else:
            self._alpha = 52
            self._beta = 320
            self._d = 0.89

    def gen_piano_roll(self, out_path):
        x, fs, _ = wavread(path)

