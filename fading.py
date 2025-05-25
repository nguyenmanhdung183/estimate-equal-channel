import numpy as np
from scipy.signal import convolve

def db_to_linear(db):
    return 10 ** (db / 10)

def generate_tdlb_channel(sampling_rate_hz):
    delays_ns = np.array([0, 100, 200, 300, 500, 800, 1100, 1600, 2300])  # ns
    powers_db = np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.9, -14.4])
    
    powers_lin = db_to_linear(powers_db)
    taps = np.sqrt(powers_lin / 2) * (np.random.randn(len(powers_db)) + 1j * np.random.randn(len(powers_db)))
    
    delays_samples = np.round(delays_ns * 1e-9 * sampling_rate_hz).astype(int)
    max_delay = delays_samples.max() + 1

    h = np.zeros(max_delay, dtype=complex)
    for i, d in enumerate(delays_samples):
        h[d] += taps[i]

    # Chuẩn hóa tổng công suất
    h /= np.linalg.norm(h)

    return h

def apply_fading_channel(signal, channel_response):
    return convolve(signal, channel_response, mode='full')  # cho kết quả chính xác hơn