import numpy as np
from scipy import signal, fftpack, stats

def rms(x):
    return np.sqrt(np.mean(x*x))

def thd(signal_v, fs, fund=50.0, n_harmonics=10):
    N = len(signal_v)
    freq = np.fft.rfftfreq(N, 1/fs)
    S = np.abs(np.fft.rfft(signal_v))
    fund_idx = np.argmin(np.abs(freq - fund))
    fund_power = S[fund_idx]**2
    harm_power = 0.0
    for h in range(2, n_harmonics+1):
        idx = np.argmin(np.abs(freq - h*fund))
        harm_power += S[idx]**2
    if fund_power == 0:
        return 0.0
    return np.sqrt(harm_power / fund_power)

def spectral_centroid(sig, fs):
    N = len(sig)
    freqs = np.fft.rfftfreq(N, 1/fs)
    S = np.abs(np.fft.rfft(sig))
    if S.sum() == 0:
        return 0.0
    return np.sum(freqs * S) / np.sum(S)

def zero_crossing_rate(x):
    return ((x[:-1] * x[1:]) < 0).sum() / len(x)

def extract_features(v, i, fs=5000, fund=50.0):
    feats = {}

    feats['v_rms'] = rms(v)
    feats['i_rms'] = rms(i)
    feats['v_mean'] = np.mean(v)
    feats['i_mean'] = np.mean(i)
    feats['v_peak'] = np.max(np.abs(v))
    feats['i_peak'] = np.max(np.abs(i))
    feats['v_kurt'] = stats.kurtosis(v)
    feats['i_kurt'] = stats.kurtosis(i)
    feats['v_zcr'] = zero_crossing_rate(v)
    feats['i_zcr'] = zero_crossing_rate(i)

    feats['v_thd'] = thd(v, fs, fund)
    feats['i_thd'] = thd(i, fs, fund)
    feats['v_centroid'] = spectral_centroid(v, fs)
    feats['i_centroid'] = spectral_centroid(i, fs)

    N = len(v)
    freqs = np.fft.rfftfreq(N, 1/fs)
    V = np.abs(np.fft.rfft(v))
    I = np.abs(np.fft.rfft(i))

    def band_energy(S, freqs, f_low, f_high):
        idx = np.where((freqs>=f_low) & (freqs<=f_high))[0]
        if len(idx)==0: return 0.0
        return np.sum(S[idx]**2) / len(idx)

    feats['v_band_low'] = band_energy(V, freqs, 0.1, fund-5)
    feats['v_band_fund'] = band_energy(V, freqs, fund-5, fund+5)
    feats['v_band_high'] = band_energy(V, freqs, fund+5, fs/2)
    feats['i_band_low'] = band_energy(I, freqs, 0.1, fund-5)
    feats['i_band_fund'] = band_energy(I, freqs, fund-5, fund+5)
    feats['i_band_high'] = band_energy(I, freqs, fund+5, fs/2)

    return feats

def features_from_window(window, fs=5000):
    v = window[0]
    i = window[1]
    feats = extract_features(v, i, fs)
    keys = sorted(feats.keys())
    vec = np.array([feats[k] for k in keys], dtype=np.float32)
    return vec, keys
