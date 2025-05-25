import numpy as np
def equalize_zf(Y, H_hat):
    """
    Cân bằng kênh sử dụng thuật toán Zero Forcing (ZF).
    
    Tham số:
    Y (numpy.ndarray): Tín hiệu nhận được (vector hoặc ma trận).
    H_hat (numpy.ndarray): Ước lượng kênh (vector).
    
    Kết quả:
    numpy.ndarray: Tín hiệu sau cân bằng.
    
    Lỗi:
    ValueError: Nếu H_hat chứa giá trị 0.
    """
    H_hat = H_hat.reshape(-1, 1)  # Đảm bảo H_hat là vector cột
    if np.any(H_hat == 0):
        raise ValueError("Ước lượng kênh H_hat chứa giá trị 0, không thể thực hiện ZF.")
    return Y / H_hat


def equalize_mmse(Y, H_hat, snr_db):
    """
    Cân bằng kênh sử dụng thuật toán MMSE.
    
    Tham số:
    Y (numpy.ndarray): Tín hiệu nhận được (vector hoặc ma trận).
    H_hat (numpy.ndarray): Ước lượng kênh (vector).
    snr_db (float): SNR (dB).
    
    Kết quả:
    numpy.ndarray: Tín hiệu sau cân bằng.
    """
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(Y) ** 2)
    noise_power = signal_power / snr_linear
    
    # MMSE: X_hat = Y * H_hat^* / (|H_hat|^2 + sigma_n^2)
    H_hat_conj = np.conj(H_hat)
    denominator = np.abs(H_hat) ** 2 + noise_power
    return Y * H_hat_conj / denominator