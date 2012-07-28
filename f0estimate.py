from __future__ import division
import os
import argparse
import numpy as np
from scikits.audiolab import wavread
from scipy.signal import get_window

from utilities import nextpow2

# set up command line argument structure
parser = argparse.ArgumentParser(description='Estimate the pitches in an audio file.')
parser.add_argument('-fin', '--filein', help='input file')
parser.add_argument('-fout', '--fileout', help='output file')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')

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
            _, n_channels = x.shape
            x = x.sum(axis=1)/n_channels

        X = self._stft(x, fs)

        # Section 2.1 Spectrally whiten the signal to suppress timbral information
        Y = self._spectral_whitening(X, fs)

        # perform iterative estimation of the fundamental frequencies in the audio file
        self._iterative_est(Y)

    def _stft(self, x, fs):
        '''
        Calculate short time fourier transform on signal that is
        hann windowed and zero-padded to twice its length.
        Hopsize = window length
        '''

        frame_len_samps = int(fs * self._frame_len_sec)
        win = get_window(self._window_func, frame_len_samps)

        # zero-pad to twice the length of the frame
        K = int(nextpow2(2*frame_len_samps))
        X = np.array([np.fft.fft(win*x[i:i+frame_len_samps], K) for i in xrange(0, len(x)-frame_len_samps, frame_len_samps)])

        return X

    def _spectral_whitening(self, X, fs, nu=0.33):
        '''
        Spectrally flatten ('whiten') the given input signal in the frequency domain, 
        with the intention of supressing timbral information.

        PARAMETERS
        ----------
        X (T, K): frequency domain input signal with T frames and FFT of length K
        fs: sampling rate of the input signal
        nu (float): amount of spectral whitening
        '''

        K = X.shape[1]

        # calculate centre frequencies c_b (Hz) of the first 30 subbands on the critical-band scale
        # c_b = 229 * (10^[(b+1)/21.4]-1)
        # this will effectively suppress all frequencies above 6935.11Hz
        # calculate one subband below and above the range to get the head and tail
        # frequencies of the triangle windows
        c = np.array([229*(10**((b+1)/21.4)-1) for b in range(0,32)])
        c_bins = np.asarray(np.floor(c*K/fs) + 1, np.int)

        # nyquist rate is half of the number of bins
        nyquist = K>>1

        # subband compression coefficients -> gamma (K/2,)
        gamma = np.zeros(nyquist)

        # for each subband
        for b in range(1,len(c_bins)-1):
            H = np.zeros(nyquist)

            left = c_bins[b-1]
            centre = c_bins[b]
            right = c_bins[b+1]

            # construct the triangular power response for each subband
            H[left:centre+1] = np.linspace(0, 1, centre - left + 1)
            H[centre:right+1] = np.linspace(1, 0, right - centre + 1)

            gamma[centre] = np.sqrt((2/K)*np.sum(H*(np.abs(X[:,:nyquist])**2)))**(nu-1)
    
            # interpolate between the previous centre bin and the current centre bin
            gamma[left:centre] = np.linspace(gamma[left], gamma[centre], centre - left)

        # calculate the whitened spectrum. Only need to store half the spectrum for analysis
        # since the bin energy is symmetric about the nyquist frequency
        Y = gamma * X[:,:nyquist]

        return Y

    def _iterative_est(self, Y):
        pass

if __name__ == '__main__':
    # parse command line arguments
    args = parser.parse_args()

    input_path = args.filein
    if not os.path.exists(input_path):
        raise ValueError('The input file does not exist')

    output_path = args.fileout

    # check file extensions are correct for this type of conversion
    _, input_ext = os.path.splitext(input_path)
    if input_ext != '.wav':
        raise ValueError('Input path must be a wav file')
    _, output_ext = os.path.splitext(output_path)
    if output_ext != '.mei':
        raise ValueError('Ouput path must have the file extension .mei')

    freq_est = F0Estimate(input_path)
    freq_est.gen_piano_roll(output_path)
