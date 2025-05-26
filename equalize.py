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
    MMSE equalizer áp dụng cho toàn bộ grid (ma trận).
    Y: ma trận tín hiệu nhận (num_subcarriers x num_symbols)
    H_hat: vector ước lượng kênh (num_subcarriers,)
    snr_db: giá trị SNR (dB)
    Trả về:
    - X_hat: tín hiệu sau cân bằng (ma trận cùng kích thước với Y)
    """
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_linear  # vì tín hiệu đã chuẩn hóa công suất = 1

    H_hat = H_hat.reshape(-1, 1)  # reshape thành vector cột (3276, 1)
    H_hat_conj = np.conj(H_hat)
    denominator = np.abs(H_hat) ** 2 + noise_power  # shape (3276, 1)

    # Broadcasting tự động với Y (3276, 14)
    X_hat = (H_hat_conj / denominator) * Y

    return X_hat
