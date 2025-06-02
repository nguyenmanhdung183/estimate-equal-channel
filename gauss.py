import numpy as np

def add_awgn_0(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # chia 2 vì là phức

    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


def add_awgn(signal, snr_db):
    """
    Thêm nhiễu AWGN vào tín hiệu phức.
    
    Tham số:
    - signal: Tín hiệu đầu vào (mảng phức).
    - snr_db: SNR tính theo dB.
    
    Trả về:
    - noisy_signal: Tín hiệu đầu vào cộng thêm nhiễu AWGN.
    """
    # Tính công suất tín hiệu (tính cả phần thực và phần ảo)
    signal_power = np.mean(np.abs(signal)**2)
    
    # Tính công suất nhiễu cần thêm vào từ SNR
    noise_power_db = 10 * np.log10(signal_power) - snr_db
    noise_power = 10 ** (noise_power_db / 10)
    
    # Tạo nhiễu Gaussian với phương sai = noise_power cho cả phần thực và phần ảo
    noise_real = np.sqrt(noise_power / 2) * np.random.randn(len(signal))  # Phần thực
    noise_imag = np.sqrt(noise_power / 2) * np.random.randn(len(signal))  # Phần ảo
    
    # Nhiễu phức
    noise = noise_real + 1j * noise_imag
    
    # Tín hiệu bị nhiễu
    noisy_signal = signal + noise
    
    return noisy_signal