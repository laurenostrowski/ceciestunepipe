from scipy import signal
import numpy as np

from ceciestunepipe.util.sound import rosa


def make_butter_bandpass(s_f: float, lo_cut:float, hi_cut:float, order: int=5):
    nyq = 0.5 * s_f
    low = lo_cut / nyq
    high = hi_cut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return {'b': b, 'a': a}

def apply_butter_bandpass(x: np.array, pars: dict, axis: int=0):
    return signal.filtfilt(pars['b'], pars['a'], x, axis=axis)

def spectrogram_db_cut(s, db_cut=65, log_scale=False):
    specgram = np.copy(s)
    max_specgram = np.max(specgram)
    # do the cut_off. Values are amplitude, not power, hence db = -20*log(V/V_0)
    if log_scale:
        # threshold = pow(10, max_specgram) * pow(10, -db_cut*0.05)
        # specgram[specgram < np.log10(threshold)] = np.log10(threshold)
        log10_threshhold = max_specgram - db_cut * 0.05
        specgram[specgram < log10_threshhold] = log10_threshhold
        # specgram /= specgram.max()  # volume normalize to max 1
    else:
        threshold = max_specgram * pow(10, -db_cut * 0.05)
        specgram[specgram < threshold] = threshold  # set anything less than the threshold as the threshold
    return specgram

def pretty_spectrogram(x, s_f, log=True, fft_size=512, step_size=64, window=None,
                       db_cut=65,
                       f_min=0.,
                       f_max=None,
                       plot=False,
                       ax=None):
    # db_cut=0 for no_trhesholding

    if window is None:
        #window = sg.windows.hann(fft_size, sym=False)
        window = ('tukey', 0.25)

    f, t, specgram = signal.spectrogram(x, fs=s_f, window=window,
                                       nperseg=fft_size,
                                       noverlap=fft_size - step_size,
                                       nfft=None,
                                       detrend='constant',
                                       return_onesided=True,
                                       scaling='spectrum',
                                       axis=-1)
                                       #mode='psd')

    if db_cut>0:
        specgram = spectrogram_db_cut(specgram, db_cut=db_cut, log_scale=False)
    if log:
        specgram = np.log10(specgram)

    if f_max is None:
        f_max = s_f/2.

    f_filter = np.where((f >= f_min) & (f < f_max))
    #return f, t, specgram

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.pcolormesh(t, f[f < f_max], specgram[f < f_max, :],
            cmap='inferno',
            rasterized=True)
        
        return f[f_filter], t, specgram[f_filter], ax

    return f[f_filter], t, specgram[f_filter]

def ms_spectrogram(x, s_f, n_window=512, step_ms=1, f_min=100, f_max=9000, cut_off=0.000055):

    # the overlap is the size of the window minus the smples in a msec
    msec_samples = int(s_f * 0.001)
    n_overlap = n_window - msec_samples * step_ms
    sigma = 1 / 200. * s_f

    # Make the spectrogram
    f, t, Sxx = signal.spectrogram(x, s_f,
                                   nperseg=n_window,
                                   noverlap=n_overlap,
                                   window=signal.gaussian(n_window, sigma),
                                   scaling='spectrum')

    if cut_off > 0:
        Sxx[Sxx < np.max((Sxx) * cut_off)] = 1
    
    Sxx[f<f_min, :] = 1

    return f[(f>f_min) & (f<f_max)], t, Sxx[(f>f_min) & (f<f_max)]

def rosa_spectrogram(y, hparams):
    D = rosa._stft(rosa.preemphasis(y,hparams), hparams)
    S = rosa._amp_to_db(np.abs(D)) - hparams['ref_level_db']
    return rosa._normalize(S, hparams)

def inv_spectrogram(spectrogram, hparams):
    '''Converts spectrogram to waveform using librosa'''
    S = rosa._db_to_amp(rosa._denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # Convert back to linear
    return rosa.inv_preemphasis(rosa._griffin_lim(S ** hparams['power'], hparams), hparams)          # Reconstruct phase

    