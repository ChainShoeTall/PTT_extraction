from scipy.signal import periodogram
import numpy as np
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
    NyquistF = fps/2.
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

    """
    Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

    This method use the Welch's method to estimate the spectral density of the BVP signal,
    then it chooses as BPM the maximum Amplitude frequency.
    """
    if pleth1.shape[0] == 0 or pleth2.shape[0] == 0:
        return np.float32(0.0)
    # if len(pleth1.shape) == 1:
    #     pleth1 = np.expand_dims(pleth1, axis=0)
    # if len(pleth2.shape) == 1:
    #     pleth2 = np.expand_dims(pleth2, axis=0)
        
    NyquistF = fps2/2.
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)
    Pfreqs, Power = periodogram(pleth2, fs=fps2, window='hamming', nfft=nfft, scaling='spectrum')
    Pfreqs = Pfreqs*60
    Pmax = np.argmax(Power)  # power max
    return SNR_sqi(pleth=pleth1, fps=fps1, reference_hr=Pfreqs[Pmax])