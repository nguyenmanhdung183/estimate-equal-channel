import numpy as np
def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # chia 2 vì là phức

    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise
