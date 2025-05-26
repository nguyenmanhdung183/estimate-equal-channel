import numpy as np
from scipy.interpolate import interp1d

def estimate_channel_ls(X, Y):
    """
    https://wirelesspi.com/channel-estimation-in-ofdm-systems/
    
    Ước lượng kênh truyền sử dụng thuật toán Least Squares (LSThông số:
    X (numpy.ndarray): Tín hiệu tham chiếu DMRS đã biết (complex).
    Y (numpy.ndarray): Tín hiệu nhận được tại các vị trí DMRS (complex).
    
    numpy.ndarray: Ước lượng kênh H cho toàn bộ tần số con.
    """

    valid_indices = np.where(X != 0)[0]
    if len(valid_indices) == 0:
        raise ValueError("Tín hiệu DMRS (X) chỉ chứa giá trị 0, không thể ước lượng kênh.")
    
    H_valid = Y[valid_indices] / X[valid_indices]
    # dùng phép nộisuy tuyến tính để ước lượng kênh cho toàn bộ tần số con
    interp_func = interp1d(valid_indices, H_valid, kind='linear', fill_value='extrapolate')
    H = interp_func(np.arange(len(X)))
    
    return H



def estimate_channel_mmse(X, Y, snr_db):
    valid_indices = np.where(X != 0)[0]
    
    snr_linear = 10 ** (snr_db / 10)
    pilot_power = np.mean(np.abs(X[valid_indices]) ** 2)
    noise_power = pilot_power / snr_linear

    # MMSE estimator đơn giản cho từng pilot
    #H_hat_mmse_valid = (np.conj(X[valid_indices]) * Y[valid_indices]) / (np.abs(X[valid_indices]) ** 2 + noise_power)

    # LS estimate tại các vị trí pilot
    H_ls_valid = Y[valid_indices] / X[valid_indices]

    # Áp dụng ước lượng MMSE 
    H_hat_mmse_valid = H_ls_valid * (np.abs(X[valid_indices]) ** 2) / (np.abs(X[valid_indices]) ** 2 + noise_power)



    # Nội suy cho toàn bộ dải tần
    from scipy.interpolate import interp1d
    interp_func = interp1d(valid_indices, H_hat_mmse_valid, kind='linear', fill_value='extrapolate')
    H_hat = interp_func(np.arange(len(X)))

    return H_hat
