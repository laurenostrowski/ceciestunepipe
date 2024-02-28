import logging
import librosa
import librosa.filters
import numpy as np

from scipy import signal
from scipy.signal import butter, lfilter

logger = logging.getLogger('ceciestunepipe.util.sound.rosa')

# For making inversible spectrograms and inverting spectrograms

def preemphasis(x,hparams):
  return signal.lfilter([1, -hparams['preemphasis']], [1], x)

def inv_preemphasis(x, hparams):
  return signal.lfilter([1], [1, -hparams['preemphasis']], x)

def melspectrogram(y,hparams,_mel_basis):
  D = _stft(preemphasis(y, hparams), hparams)
  S = _amp_to_db(_linear_to_mel(np.abs(D),_mel_basis)) - hparams['ref_level_db']
  return _normalize(S, hparams)

def find_endpoint(wav, hparams, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams['sample_rate'] * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    taken from Tim Sainburg or Marvin Theilk
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    x = np.hstack((X, append))

    valid = len(x) - window_size
    nw = valid // window_step
    out = np.ndarray((nw, window_size), dtype=x.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * window_step
        stop = start + window_size
        out[i] = x[start: stop]
    return out

def _griffin_lim(S, hparams):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles,hparams)
  for i in range(hparams['griffin_lim_iters']):
    angles = np.exp(1j * np.angle(_stft(y, hparams)))
    y = _istft(S_complex * angles, hparams)
  return y

def _stft(y, hparams):
  n_fft, hop_length, win_length = _stft_parameters(hparams)
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y, hparams):
  _, hop_length, win_length = _stft_parameters(hparams)
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _stft_parameters(hparams):
  n_fft = (hparams['num_freq'] - 1) * 2
  hop_length = int(hparams['frame_shift_ms'] / 1000 * hparams['sample_rate'])
  win_length = int(hparams['frame_length_ms'] / 1000 * hparams['sample_rate'])
  return n_fft, hop_length, win_length

def _linear_to_mel(spectrogram, _mel_basis):
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(hparams):
  n_fft = (hparams['num_freq'] - 1) * 2
  return librosa.filters.mel(hparams['sample_rate'], n_fft, n_mels=hparams['num_mels'], fmin = hparams['fmin'], fmax=hparams['fmax'])

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _normalize(S, hparams):
  return np.clip((S - hparams['min_level_db']) / -hparams['min_level_db'], 0, 1)

def _denormalize(S, hparams):
  return (np.clip(S, 0, 1) * -hparams['min_level_db']) + hparams['min_level_db']