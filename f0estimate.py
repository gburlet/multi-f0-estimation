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
            self._min_f0 = 65

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

        # perform iterative estimation of the fundamental periods in the audio file
        f0_estimations = self._iterative_est(Y, fs)

        return f0_estimations

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

        T, K = X.shape
        nyquist_freq = fs/2
        nyquist_bins = K>>1

        # calculate centre frequencies c_b (Hz) of subbands on the critical-band scale
        # c_b = 229 * (10^[(b+1)/21.4]-1)
        # calculate one subband below and above the range to get the head and tail
        # frequencies of the triangle windows
        c = []      # centre frequencies of critical-bands
        b = 0       # critical band index
        while True:
            centre_freq = 229*(10**((b+1)/21.4)-1)
            if centre_freq < nyquist_freq:
                c.append(centre_freq)
                b += 1
            else:
                break

        c = np.asarray(c)
        c_bins = np.asarray(np.floor(c*K/fs) + 1, np.int)

        # subband compression coefficients -> gamma (K/2,)
        gamma = np.zeros([T, nyquist_bins])

        # for each subband
        for b in xrange(1,len(c_bins)-1):
            H = np.zeros(nyquist_bins)

            left = c_bins[b-1]
            centre = c_bins[b]
            right = c_bins[b+1]

            # construct the triangular power response for each subband
            H[left:centre+1] = np.linspace(0, 1, centre - left + 1)
            H[centre:right+1] = np.linspace(1, 0, right - centre + 1)

            # multiply by 2, since energy is symmetric about the nyquist rate
            gamma[:,centre] = np.sqrt((2/K)*np.sum(H*(np.abs(X[:,:nyquist_bins])**2), axis=1))**(nu-1)
    
            # interpolate between the previous centre bin and the current centre bin
            # for each STFT frame
            for t in xrange(T):
                gamma[t,left:centre] = np.linspace(gamma[t,left], gamma[t,centre], centre - left)

        # calculate the whitened spectrum. Only need to store half the spectrum for analysis
        # since the bin energy is symmetric about the nyquist frequency
        Y = gamma * X[:,:nyquist_bins]

        return Y

    def _iterative_est(self, Y, fs):
        f0_estimations = []

        T = Y.shape[0]
        # for each STFT frame
        for t in xrange(T):
            f0_frame_estimations = []
            # TODO: while there are still f0's to search for
            tau_hat = self._search_smax(Y[t,:], fs, tau_prec=1.0)
            f0_frame_estimations.append(fs/tau_hat)
            f0_estimations.append({'f0s': f0_frame_estimations, 'ts': t*self._frame_len_sec})

        return f0_estimations

    def _search_smax(self, Y_t, fs, tau_prec=0.5):
        Q = 0           # index of the new block
        q_best = 0      # index of the best block
       
        tau_low = [round(fs/self._max_f0)] # in samples/cycle
        tau_up = [round(fs/self._min_f0)]  # in samples/cycle
        smax = [0]

        while tau_up[q_best] - tau_low[q_best] > tau_prec:
            # split the best block and compute new limits
            Q += 1
            tau_low.append((tau_low[q_best] + tau_up[q_best])/2)
            tau_up.append(tau_up[q_best])
            tau_up[q_best] = tau_low[Q]

            # compute new saliences for the two block-halves
            for q in [q_best, Q]:
                salience = self._calc_salience(Y_t, fs, tau_low[q], tau_up[q])
                if q == q_best:
                    smax[q_best] = salience
                else:
                    smax.append(salience)

            q_best = np.argmax(smax)

        tau_hat = (tau_low[q_best] + tau_up[q_best])/2

        return tau_hat

    def _calc_salience(self, Y_t, fs, tau_low, tau_up):
        salience = 0

        tau = (tau_low + tau_up)/2
        delta_tau = tau_up - tau_low

        # calculate the number of harmonics under the nyquist frequency
        # the statement below is equivalent to floor((fs/2)/fo)
        num_harmonics = int(np.floor(tau/2))

        # calculate all harmonic weights
        harmonics = np.arange(num_harmonics)+1
        g = (fs/tau_low + self._alpha) / (harmonics*fs/tau_up + self._beta)

        # calculate lower and upper bounds of partial vicinity
        nyquist_bin = len(Y_t)
        K = nyquist_bin<<1
        lb_vicinity = K/(tau + delta_tau/2)
        ub_vicinity = K/(tau - delta_tau/2)

        # for each harmonic
        for m in xrange(1,num_harmonics+1):
            harmonic_lb = round(m*lb_vicinity)
            harmonic_ub = min(round(m*ub_vicinity), nyquist_bin)
            max_vicinity = np.max(np.abs(Y_t[harmonic_lb-1:harmonic_ub]))

            salience += g[m-1] * max_vicinity

        return salience

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
    f0_estimates = freq_est.gen_piano_roll(output_path)
    print f0_estimates
