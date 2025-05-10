import numpy as np
import math
from scipy.signal.windows import hann
from scipy.signal import butter, filtfilt

def CHROM_win(signal, FS=30, WinSec=1.6, LPF=0.7, HPF=4.0):
    ""
    """
    CHROM method on CPU using Numpy. (adapted from rPPG-toolbox)

    signal: [n_frame, n_estimator, n_channel]

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. 
    IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """    

    FN, n_est, _ = signal.shape
    NyquistF = 1/2*FS
    B, A = butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)
    if(WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))

    totallen = (WinL//2)*(NWin+1)
    S = np.zeros((n_est, totallen))
    for i_est in range(n_est):
        WinS = 0
        WinM = int(WinS+WinL//2)
        WinE = WinS+WinL
        for i in range(NWin):
            RGBBase = np.mean(signal[WinS:WinE, i_est, :], axis=0)
            RGBNorm = np.zeros((WinE-WinS, 3))
            for temp in range(WinS, WinE):
                RGBNorm[temp-WinS] = np.true_divide(signal[temp, i_est, :], RGBBase)
            Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
            Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
            Xf = filtfilt(B, A, Xs, axis=0)
            Yf = filtfilt(B, A, Ys)

            Alpha = np.std(Xf) / np.std(Yf)
            SWin = Xf-Alpha*Yf
            SWin = np.multiply(SWin, hann(WinL))

            temp = SWin[:int(WinL//2)]
            S[i_est, WinS:WinM] = S[i_est, WinS:WinM] + SWin[:int(WinL//2)]
            S[i_est, WinM:WinE] = SWin[int(WinL//2):]
            WinS = WinM
            WinM = WinS+WinL//2
            WinE = WinS+WinL
    return S

def POS_win(signal, fps=60):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * fps)   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H