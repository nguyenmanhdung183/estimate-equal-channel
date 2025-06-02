import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz

def estimate_channel_ls(X, Y): # X là M*1, Y là M*1
    """
    https://hal.science/hal-02268202/document
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
    interp_func = interp1d(valid_indices, H_valid, kind='cubic', fill_value='extrapolate')
    H_hat = interp_func(np.arange(len(X)))
    
    return H_hat


def estimate_channel_mmse(X, Y, snr_db, rho=0.8):
    """
    Ước lượng kênh MMSE sử dụng ma trận hiệp phương sai kênh R_H.

    X: mảng 1 chiều pilot truyền (M×1)
    Y: mảng 1 chiều tín hiệu nhận tại pilot (M×1)
    snr_db: SNR tính theo dB
    rho: hệ số tạo ma trận hiệp phương sai kênh (tùy chọn)
    
    Trả về:
    H_hat: vector ước lượng kênh MMSE (M×1)
    """
    N = len(X)  # Số lượng subcarrier
    # Tạo ma trận hiệp phương sai kênh (R_H)
    R_H = rho ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    
    valid_indices = np.where(X != 0)[0]  # Tìm vị trí của các pilot
    if len(valid_indices) == 0:
        raise ValueError("Tín hiệu pilot rỗng hoặc toàn 0.")

    X_valid = X[valid_indices]
    Y_valid = Y[valid_indices]
    R_H_valid = R_H[np.ix_(valid_indices, valid_indices)]  # Lọc ma trận hiệp phương sai cho các pilot

    # Ma trận chéo pilot
    X_p = np.diag(X_valid)

    # Chuyển đổi SNR từ dB sang giá trị tuyến tính
    snr_linear = 10 ** (snr_db / 10)
    
    # Công suất trung bình của pilot
    pilot_power = np.mean(np.abs(X_valid)**2)
    
    # Phương sai nhiễu
    noise_power = pilot_power / snr_linear

    # Ước lượng LS (Least Squares)
    H_ls = np.linalg.inv(X_p) @ Y_valid  # Hoặc H_ls = Y_valid / X_valid

    # Tính nghịch đảo (X_p X_p^H)⁻¹ = (|X_p|^2)⁻¹ vì X_p là ma trận chéo
    XpXpH_inv = np.diag(1 / (np.abs(X_valid)**2))

    # Tính ma trận trong ngoặc
    matrix = R_H_valid + noise_power * XpXpH_inv

    # Ước lượng MMSE
    H_mmse = R_H_valid @ np.linalg.inv(matrix) @ H_ls

    # Trả về vector kích thước ban đầu, nội suy cho các vị trí không phải pilot nếu cần
    H_hat = np.zeros_like(X, dtype=complex)
    H_hat[valid_indices] = H_mmse

    # Nội suy tuyến tính cho các subcarrier không phải pilot (nếu có)
    from scipy.interpolate import interp1d
    if len(valid_indices) < len(X):
        interp_func = interp1d(valid_indices, H_mmse, kind='cubic', fill_value='extrapolate')
        all_indices = np.arange(len(X))
        H_hat = interp_func(all_indices)

    return H_hat

def estimate_channel_mmse_1(X, Y, snr_db, rho = 0.8):
    """
    Ước lượng kênh MMSE sử dụng ma trận hiệp phương sai kênh R_H.
    
    X: mảng 1 chiều pilot truyền (M×1)
    Y: mảng 1 chiều tín hiệu nhận tại pilot (M×1)
    snr_db: SNR tính theo dB
    R_H: ma trận hiệp phương sai kênh (M×M)
    
    Trả về:
    H_hat: vector ước lượng kênh MMSE (M×1)
    """
    N = len(X)
    rho = 0.99
    R_H = rho ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    
    valid_indices = np.where(X != 0)[0]
    if len(valid_indices) == 0:
        raise ValueError("Tín hiệu pilot rỗng hoặc toàn 0.")

    X_valid = X[valid_indices]
    Y_valid = Y[valid_indices]
    R_H_valid = R_H[np.ix_(valid_indices, valid_indices)]

    # Ma trận chéo pilot
    X_p = np.diag(X_valid)

    # Chuyển dB sang giá trị tuyến tính
    snr_linear = 10 ** (snr_db / 10)
    
    # Công suất pilot trung bình
    pilot_power = np.mean(np.abs(X_valid)**2)
    
    # Phương sai nhiễu
    noise_power = pilot_power / snr_linear

    # Ước lượng LS
    H_ls = np.linalg.inv(X_p) @ Y_valid  # hoặc H_ls = Y_valid / X_valid
    
    # Tính nghịch đảo (X_p X_p^H)⁻¹ = (|X_p|^2)⁻¹ vì X_p là ma trận chéo
    XpXpH_inv = np.diag(1 / (np.abs(X_valid)**2))

    # Tính ma trận trong ngoặc
    matrix = R_H_valid + noise_power * XpXpH_inv

    # Tính ước lượng MMSE
    H_mmse = R_H_valid @ np.linalg.inv(matrix) @ H_ls

    # Trả về vector kích thước ban đầu, nội suy cho các vị trí không phải pilot nếu cần
    H_hat = np.zeros_like(X, dtype=complex)
    H_hat[valid_indices] = H_mmse

    # Nội suy tuyến tính cho các subcarrier không phải pilot (nếu có)
    from scipy.interpolate import interp1d
    if len(valid_indices) < len(X):
        interp_func = interp1d(valid_indices, H_mmse, kind='cubic', fill_value='extrapolate')
        all_indices = np.arange(len(X))
        H_hat = interp_func(all_indices)

    return H_hat,  pilot_power


def estimate_channel_mmse_0(X, Y, snr_db, rho=0.8):
    """
    Uớc lượng kênh MMSE với ma trận hiệp phương sai kênh chính xác.

    X: mảng tín hiệu pilot truyền
    Y: mảng tín hiệu nhận tại pilot
    snr_db: SNR tính bằng dB
    df: Khoảng cách tần số giữa các subcarrier
    tau_rms: Độ trễ lan tỏa RMS của kênh

    Trả về:
    H_hat: vector uớc lượng kênh MMSE
    """
    
    # Các tham số của kênh
    df = 100e6 / 3276  # Khoảng cách tần số giữa các subcarrier
    tau_rms = 1e-7     # Độ trễ lan tỏa RMS của kênh
    
    # Số lượng tín hiệu pilot
    N = len(X)
    valid_indices = np.where(X != 0)[0]  # Tìm chỉ mục các giá trị không bằng 0 trong X
    N_pilots = len(valid_indices)
    
    # Tạo ma trận hiệp phương sai kênh (R_H_valid)
    lags = np.arange(0, N_pilots)
    R_f_pos = 1 / (1 + 1j * 2 * np.pi * lags * df * tau_rms)
    first_row = R_f_pos
    first_col = np.conj(first_row[::-1])  # Lấy đối xứng phức của hàng đầu tiên
    R_H_valid = toeplitz(first_row, first_col)  # Tạo ma trận Toeplitz
    
    # Lọc các giá trị hợp lệ của X và Y
    X_valid = X[valid_indices]
    Y_valid = Y[valid_indices]
    
    # Ma trận chéo của X (chứa các bình phương của tín hiệu pilot)
    XpXpH_inv = np.diag(1 / np.abs(X_valid) ** 2)
    
    # Công suất của tín hiệu pilot
    pilot_power = np.mean(np.abs(X_valid) ** 2)
    
    # Chuyển SNR từ dB sang giá trị tuyến tính
    snr_linear = 10 ** (snr_db / 10)
    
    # Tính phương sai nhiễu
    noise_power = pilot_power / snr_linear
    
    # Ước lượng kênh Least Squares (LS)
    H_ls = Y_valid / X_valid
    
    # Tính ma trận trong công thức MMSE
    matrix = R_H_valid + noise_power * XpXpH_inv
    
    # Đảm bảo ma trận có thể đảo ngược
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inv_matrix = np.linalg.pinv(matrix)  # Sử dụng Pseudo-inverse nếu ma trận không khả nghịch
    
    # Ước lượng kênh MMSE
    H_mmse = np.dot(R_H_valid, np.dot(inv_matrix, H_ls))
    
    # Khởi tạo vector kênh ước lượng với các giá trị mặc định
    H_hat = np.zeros_like(X, dtype=complex)
    H_hat[valid_indices] = H_mmse
    
    # Chia các tín hiệu thành các nhóm tương ứng với các biểu tượng thời gian có pilot
    symbol_length = len(valid_indices)
    num_symbols = N // symbol_length
    
    # Tìm biểu tượng có pilot
    pilot_symbol = None
    for m in range(num_symbols):
        if np.array_equal(valid_indices, np.arange(m * symbol_length, (m + 1) * symbol_length)):
            pilot_symbol = m
            break
    
    if pilot_symbol is None:
        raise ValueError("Không tìm thấy biểu tượng có pilot")
    
    # Lấy các giá trị kênh cho biểu tượng pilot và sao chép cho tất cả các biểu tượng
    H_pilot = H_hat[pilot_symbol * symbol_length : (pilot_symbol + 1) * symbol_length]
    for m in range(num_symbols):
        H_hat[m * symbol_length : (m + 1) * symbol_length] = H_pilot
    
    return H_hat
