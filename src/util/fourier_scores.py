"""Match the installed Waterfall Fourier implementation, including legacy scores."""
import numpy as np
from scipy.fft import rfft


def fourier_scores(dense, watermarking_fn, legacy=False):
    wf = watermarking_fn
    # 0.3.4 exposes num_fns/max_freq and fixes both the sine sign and odd N.
    # Keep the historical 0.2.13 computation byte-for-byte for Llama results.
    if hasattr(wf, "num_fns") and not legacy:
        f = rfft(dense, axis=-1)
        f = (f[:, 1:-1] if wf.N % 2 == 0 else f[:, 1:]).astype(np.complex64)
        return np.concatenate((f.real, -f.imag), axis=1) * wf.scaling_factor
    f = rfft(dense, axis=-1)[:, 1:-1].astype(np.complex64)
    return np.concatenate((f.real, f.imag), axis=1) * wf.scaling_factor
