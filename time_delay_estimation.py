import numpy as np

def frac_delay(x, y, fs, **kwargs):
    """
    Fractional delay estimation using parabolic interpolation
    Args:
        x, y: input signals
        fs: sampling frequency
        kwargs: additional parameters (not used in this method)
    Returns:
        delay in seconds
    """
    # Compute cross-correlation (full mode to get negative and positive lags)
    corr = np.correlate(x, y, mode='full')  
    lags = np.arange(-len(y)+1, len(x))    # corresponding lag values

    # Find index of max correlation
    k0 = np.argmax(corr)
    peak_lag = lags[k0]        # integer lag of max correlation

    # Parabolic interpolation around the peak
    if 0 < k0 < len(corr)-1:
        y_m1, y_0, y_p1 = corr[k0-1], corr[k0], corr[k0+1]
        denom = (y_m1 - 2*y_0 + y_p1)
        if denom != 0:
            delta = (y_m1 - y_p1) / (2 * denom)
        else:
            delta = 0.0
    else:
        delta = 0.0
    frac_delay = (peak_lag + delta) / fs   # convert to seconds
    return frac_delay

def sinc_interp(x, y, fs, **kwargs):
    """
    Sinc interpolation based delay estimation
    Args:
        x, y: input signals
        fs: sampling frequency
        kwargs: 
            scale_factor (int): upsampling factor (default=10)
    Returns:
        delay in seconds
    """
    scale_factor = kwargs.get('scale_factor', 10)
    N = len(x)
    M = scale_factor * N
    X = np.fft.rfft(x, n=N)
    Y = np.fft.rfft(y, n=N)
    cross_spec = X * np.conj(Y)

    cross_spec = np.pad(cross_spec, (0, M//2-N//2), constant_values=(0,))
    corr = np.fft.irfft(cross_spec, n=M)
    corr = np.roll(corr, M//2)  
    valid_lags = np.arange(-N+1, N)
    corr_valid = corr[M//2 - (N-1) : M//2 + (N-1) + 1]
    k0 = np.argmax(corr_valid)
    peak_lag_frac = valid_lags[0] + k0

    return (peak_lag_frac/scale_factor) / fs  # convert to seconds

def gcc_delay(x, y, fs, **kwargs):
    """
    GCC-PHAT based delay estimation
    Args:
        x, y: input signals
        fs: sampling frequency
        kwargs:
            tau_grid: array of time delays to search
            PHAT_weight (bool): whether to use PHAT weighting (default=False)
    Returns:
        delay in seconds
    """
    PHAT_weight = kwargs.get('PHAT_weight', False)
    tau_grid = kwargs.get('tau_grid', np.linspace(-.1, .1, 1000))
    
    N = len(x)
    X = np.fft.rfft(x, n=N)
    Y = np.fft.rfft(y, n=N)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    gcc_list = []
    
    for tau_ in tau_grid:
        C = X * np.conj(Y)*np.exp(1j*2*np.pi*freqs*tau_)
        if PHAT_weight:
            EPS = 1e-6
            C_phat = C / (np.abs(C) + EPS)
        else:
            C_phat = C
        gcc_phat = np.fft.irfft(C_phat, n=N)
        gcc_list.append(gcc_phat[0])
    
    return tau_grid[np.argmax(gcc_list)]  # already in seconds