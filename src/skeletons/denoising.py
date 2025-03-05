import abc
import numpy as np
import pywt

from configurable import TypedConfigurable, Schema

class BaseDenoising(TypedConfigurable):
    """
    BaseDenoising for all denoising algorithms.

    Configuration :
        - **name** (str) : The name of denoising algorithm, BaseDenoising, used.
    """
    @abc.abstractmethod
    def get_denoised_signal(self, signal):
        """
        Obtain denoised signal.

        Args:
            signal (np.array) : Signal along velocity axis.

        Returns:
            Denoised signal along velocity axis.
        """
        pass    


class NoDenoising(BaseDenoising):
    """
    Used to return original signal.
    """
    def get_denoised_signal(self, signal):
        return signal
    
class Wavelet(BaseDenoising):
    """
    Wavelket-based signal denoising using scipy.

    This method applies wavelet thresholding to remove noise from a signal.

    Configuration:
        - **noisethr** (float): Noise threshold parameter (default: 0.95).
    """
    config_schema = {
        'noisethr': Schema(float, default=0.95),
    }

    def get_denoised_signal(self, signal):
        smoothed_signal = signal.copy()
        wavelet = 'db4'  
        coeffs = pywt.wavedec(smoothed_signal, wavelet)
        sigma = np.median(np.abs(coeffs[-1])) / self.noisethr
        seuil = sigma * np.sqrt(2 * np.log(len(smoothed_signal)))
        coeffs_seuilles = [pywt.threshold(c, seuil, mode='soft') for c in coeffs]
        smoothed_signal = pywt.waverec(coeffs_seuilles, wavelet)

        return smoothed_signal
