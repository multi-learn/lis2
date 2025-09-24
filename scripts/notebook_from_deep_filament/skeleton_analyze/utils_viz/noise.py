import numpy as np

import pywt


def smooth_signal(values, noisethr):
    smoothed_signal = values.copy()
    wavelet = 'db4'  
    coeffs = pywt.wavedec(smoothed_signal, wavelet)
    sigma = np.median(np.abs(coeffs[-1])) / noisethr
    seuil = sigma * np.sqrt(2 * np.log(len(smoothed_signal)))
    coeffs_seuilles = [pywt.threshold(c, seuil, mode='soft') for c in coeffs]
    smoothed_signal = pywt.waverec(coeffs_seuilles, wavelet)

    return smoothed_signal
