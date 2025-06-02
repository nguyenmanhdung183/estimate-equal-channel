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


import numpy as np

def equalize_mmse(Y, H_hat, snr_db, signal_power=1.0):
    """
    MMSE equalizer áp dụng cho toàn bộ grid (ma trận).
    Y: ma trận tín hiệu nhận (num_subcarriers x num_symbols), phức
    H_hat: vector ước lượng kênh (num_subcarriers,), phức
    snr_db: giá trị SNR tính theo dB
    signal_power: công suất tín hiệu đầu vào (mặc định 1.0)

    Trả về:
    - X_hat: tín hiệu sau cân bằng (ma trận cùng kích thước với Y)
    """

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear  # noise variance σ²

    H_hat = H_hat.reshape(-1, 1)  # (num_subcarriers, 1)
    H_hat_conj = np.conj(H_hat)
    H_abs2 = np.abs(H_hat)**2

    # Theo công thức MMSE:
    # Q = H^H / (|H|^2 + σ² / P_x)
    denominator = H_abs2 + noise_power / signal_power

    # Apply MMSE equalizer (broadcasting với Y)
    X_hat = (H_hat_conj / denominator) * Y

    return X_hat
