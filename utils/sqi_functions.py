import scipy
import antropy
from sklearn.decomposition import PCA
from scipy.signal import periodogram
import neurokit2 as nk
import numpy as np
# from dtw import dtw, rabinerJuangStepPattern
# To be implemented:
# - Hampel filter

###============================== SQI for Timeseries ==============================###

def kurt_sqi(signal, kurtosis_method='fisher'):
    """Return the kurtosis of the signal, with Fisher's or Pearson's method.

    Parameters
    ----------
    signal : np.array
        The input signal
    kurtosis_method : str
        Compute kurtosis (ksqi) based on "fisher" (default) or "pearson" definition.

    Reference
    ----------
    Source: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_quality.py
    """
    # Fisher's definition of kurtosis (normal ==> 0); Pearson's definition of kurtosis (normal ==> 3)
    if kurtosis_method == "fisher":
        return scipy.stats.kurtosis(signal, fisher=True)
    elif kurtosis_method == "pearson":
        return scipy.stats.kurtosis(signal, fisher=False)
    
def skew_sqi(signal):
    """Return the skewness of the signal.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return scipy.stats.skew(signal)

def pur_sqi(signal):
    """ Returns the signal purity of the input. 
    In the case of a periodic signal with a single dominant frequency, 
    it takes the value of one and approaches zero for non-sinusoidal noisy signals.
    antropy.hjorth_params returns 2 floats: (mobility, complexity).
    Complexity is the value we want.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return antropy.hjorth_params(signal)[1]

def ent_sqi(signal):
    """ Returns the sample entropy of the signal.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return antropy.sample_entropy(signal)

def pca_sqi(signal):
    """ Returns a PCA sqi of the input signals.
        Defined as the sum of the first 5 singular values divided by the sum of all singular values.
    Parameters 
    ----------
    signal : array-like of shape (n_samples, n_features)
        Multivariate time-series, shape is at least 2 dimensional
    """
    # todo: Currently, we are only using single channel (1d) swis
    pca = PCA(n_components=None)
    pca.fit(signal)

    return np.sum(pca.singular_values_[:5]) / np.sum(pca.singular_values_)

def autocorr_sqi(signal, lag):
    """Calculates the autocorrelation of the specified lag, according to the formula in https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Parameters
    ----------
    signal : np.array
        The input signal
    lag : int
        The lag to use forthe autocorrelation calculation of the signal

    Reference
    ----------    
    source: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#autocorrelation
    """
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.

    if len(signal) < lag:
        raise Warning("The lag is larger than the signal length. Returning NaN.")
        return np.nan
    # Slice the relevant subseries based on the lag
    y1 = signal[: (len(signal) - lag)]
    y2 = signal[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(signal)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(signal)
    if np.isclose(v, 0):
        return np.nan
    else:
        return sum_product / ((len(signal) - lag) * v)
    
def zc_sqi(signal):
    """Returns the zero crossing rate.

    Parameters
    ----------
    signal : np.array
        The input signal
    """
    return antropy.num_zerocross(signal)

# def snr_sqi(signal_raw, signal_noise):
#     """Returns the signal to noise ratio (SNR). There are many ways to define SNR, here, we use std of filtered vs std of raw signal.

#     Parameters
#     ----------
#     signal_raw : np.array
#         Raw input signal
#     signal_cleaned : np.array    
#         Cleaned input signal
#     """
#     return np.std(np.abs(signal_raw)) / np.std(np.abs(signal_noise))

def f_sqi(signal, window_size=3, threshold=1e-5):
    """Returns an sqi that computes percentage of flatness in the signal. 
    Constant values over a longer period (flat line) may be caused by sensor failures.

    Parameters
    ----------
    signal : np.array
        The input signal
    window_size : int
        Window to detect flat line, larger values will lower detection sensitivity
    threshold : float
        Threshold of flatness. I.e. Where (max-min) is considered equivalent  

    Reference
    ----------
    Source: https://github.com/DHI/tsod/blob/main/tsod/detectors.py
    """
    if window_size >= len(signal): return 0

    rolling = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_size)
    rollmax = np.nanmax(rolling, axis=1)
    rollmin = np.nanmin(rolling, axis=1)
    
    anomalies = rollmax - rollmin < threshold
    anomalies[0] = False  # first element cannot be determined
    anomalies[-1] = False

    idx = np.where(anomalies)[0]
    if idx is not None:
        # assuming window size = 3
        # remove also points before and after each detected anomaly
        anomalies[idx[idx > 0] - 1] = True
        maxidx = len(anomalies) - 1
        anomalies[idx[idx < maxidx] + 1] = True

    return np.sum(anomalies) / len(anomalies)
def get_generic_sqis(signal, fps=30):
    """ Returns sqis for a generic signal.

    Parameters
    ----------
    signal : np.array
        The input signal

    Returns
    ----------
    List : list
        List of 10 time series sqis
    """
    return [
        kurt_sqi(signal, kurtosis_method='fisher'),
        skew_sqi(signal),
        pur_sqi(signal),
        ent_sqi(signal),
        zc_sqi(signal),
        f_sqi(signal, window_size=3, threshold=1e-7),
        np.nanmean(signal),
        np.nanstd(signal),
        np.nanmax(signal),
        np.nanmin(signal)
    ], \
    {
        'sqi_kurtosis': kurt_sqi(signal, kurtosis_method='fisher'),
        'sqi_skewness': skew_sqi(signal),
        'sqi_purity': pur_sqi(signal),
        'sqi_entropy': ent_sqi(signal),
        'sqi_zero_crossing_rate': zc_sqi(signal),
        'sqi_snr_rate': SNR_sqi_wave2wave(signal, signal, fps, fps),
        # 'sqi_flatness': f_sqi(signal, window_size=3, threshold=1e-7),
        'sqi_mean': np.nanmean(signal),
        'sqi_std': np.nanstd(signal),
        'sqi_max': np.nanmax(signal),
        'sqi_min': np.nanmin(signal),
    }

###============================== SQI for PPG ==============================###

def perfusion_sqi(pleth_raw, pleth_cleaned):
    """Returns perfusion of Pleth. The perfusion index is the ratio of the pulsatile blood flow to the nonpulsatile 
    or static blood in peripheral tissue. In other words, it is the difference of the 
    amount of light absorbed through the pulse of when light is transmitted through 
    the finger. It is calculated as AC/DC * 100

    Parameters
    ----------
    pleth_raw : np.array
        Input unfiltered Pleth signal
    pleth_cleaned : np.array
        Input filtered Pleth signal
    """
    try:
        return (np.nanmax(pleth_cleaned) - np.nanmin(pleth_cleaned)) / np.nanmean(pleth_raw) * 100
    except Exception as e:
        return np.nan

def snr_pleth_sqi(pleth_raw, pleth_ref):
    '''Computes the signal-to-noise ratio of the pleth
    signals according to the method by -- de Haan G. et al., IEEE Transactions on Biomedical Engineering (2013).
    SNR calculated as the ratio (in dB) of power contained within +/- 0.1 Hz
    of the reference heart rate frequency and +/- 0.2 of its first
    harmonic and sum of all other power between 0.5 and 4 Hz.
    Adapted from https://github.com/danmcduff/iphys-toolbox/blob/master/tools/bvpsnr.m
    '''
    pass



def SNR_sqi(pleth, fps, reference_hr):
    '''Computes the signal-to-noise ratio of the BVP
    signals according to the method by -- de Haan G. et al., IEEE Transactions on Biomedical Engineering (2013).
    SNR calculated as the ratio (in dB) of power contained within +/- 0.1 Hz
    of the reference heart rate frequency and +/- 0.2 of its first
    harmonic and sum of all other power between 0.5 and 4 Hz.
    Adapted from https://github.com/danmcduff/iphys-toolbox/blob/master/tools/bvpsnr.m
    '''
   
    interv1 = 0.1*60
    interv2 = 0.2*60
    valid_range1 = 0.5*60 
    valid_range2 = 4*60
    NyquistF = fps/2.;
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)

    pfreqs, power = periodogram(pleth, fs=fps, window='hamming', nfft=nfft, scaling='spectrum')
    pfreqs = pfreqs*60

    GTMask1 = np.logical_and(pfreqs>=reference_hr-interv1, pfreqs<=reference_hr+interv1)
    GTMask2 = np.logical_and(pfreqs>=(reference_hr*2)-interv2, pfreqs<=(reference_hr*2)+interv2)
    validMask = np.logical_and(pfreqs>=valid_range1, pfreqs<=valid_range2)
    GTMask = np.logical_and(np.logical_or(GTMask1, GTMask2), validMask)
    FMask = np.logical_and(np.logical_not(GTMask), validMask)

    SPower = np.sum(power[GTMask])
    NPower = np.sum(power[FMask])
    snr = 10*np.log10(SPower/NPower)
    return snr

def SNR_sqi_wave2wave(pleth1, pleth2, fps1, fps2):

    from pyVHR.BPM.utils import Welch
    """
    Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

    This method use the Welch's method to estimate the spectral density of the BVP signal,
    then it chooses as BPM the maximum Amplitude frequency.
    """
    if pleth1.shape[0] == 0 or pleth2.shape[0] == 0:
        return np.float32(0.0)
    # if len(pleth1.shape) == 1:
    #     pleth1 = np.expand_dims(pleth1, axis=0)
    if len(pleth2.shape) == 1:
        pleth2 = np.expand_dims(pleth2, axis=0)
        
    Pfreqs, Power = Welch(pleth2, fps2)
    Pmax = np.argmax(Power, axis=1)  # power max
    return SNR_sqi(pleth=pleth1, fps=fps1, reference_hr=Pfreqs[Pmax.squeeze()])

def get_pleth_sqis(pleth_raw, pleth_cleaned):
    """ Returns all Pleth sqis.

    Parameters
    ----------
    pleth_raw : np.array
        Input unfiltered Pleth signal
    pleth_cleaned : np.array
        Input filtered Pleth signal
    
    Returns
    ----------
    List : list
        List of 1 implemented single channel Pleth sqi and 10 time series sqis
        for a total of 11 features 
    """
    # pleth_cleaned = nk.ppg_clean(ppg_signal=pleth_raw, sampling_rate=sampling_rate, method='elgendi')
    pleth_sqis = [
        perfusion_sqi(pleth_raw, pleth_cleaned),
    ]

    generic_sqis = get_generic_sqis(signal=pleth_raw)
    return pleth_sqis + generic_sqis

# def dtw_score(pleth1, pleth2):
#     """ Returns the Dynamic Time Warping (DTW) score between two signals.

#     Parameters
#     ----------
#     pleth1 : np.array
#         Input Pleth signal 1
#     pleth2 : np.array
#         Input Pleth signal 2
#     """
#     return dtw(pleth1, pleth2)