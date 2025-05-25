'''
Gán các symbol vào các subcarriers (IFFT)

Thực hiện IFFT để đưa về miền thời gian

(Tùy chọn) Thêm Cyclic Prefix để tăng khả năng chống nhiễu đa đường

'''
import numpy as np

def ofdm_single_symbol(qam_symbols, n_fft=4096, cp_len=288):#Tần số ➝ Thời gian
    """
    Tạo 1 ký hiệu OFDM duy nhất từ 3276 QAM symbols.
    
    Params:
        qam_symbols: mảng QAM symbols (ví dụ 3276 symbol)
        n_fft: số điểm FFT (ví dụ 4096)
        cp_len: độ dài cyclic prefix (theo chuẩn thường ~7%)
    
    Returns:
        1 ký hiệu OFDM (1D numpy array, thời gian)
    """
    if len(qam_symbols) > n_fft:
        raise ValueError("Số symbol lớn hơn số subcarriers!")

    # Khởi tạo phổ tần số: n_fft subcarriers
    freq_domain = np.zeros(n_fft, dtype=complex)

    # Ánh xạ symbol vào giữa phổ (bỏ DC, padding 2 bên)
    start = (n_fft - len(qam_symbols)) // 2
    freq_domain[start:start+len(qam_symbols)] = qam_symbols

    # IFFT để đưa sang miền thời gian
    time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain))

    # Thêm cyclic prefix
    cp = time_domain[-cp_len:]
    ofdm_with_cp = np.concatenate([cp, time_domain])

    return ofdm_with_cp


'''
Cắt CP khỏi mỗi symbol
Thực hiện FFT để đưa về miền tần số
Trích xuất lại lưới tài nguyên (resource grid)
'''
def ofdm_demodulate(rx_signal, fft_size=4096, cp_len=288, num_symbols=14, num_subcarriers=3276): #Thời gian ➝ Tần số
    symbol_len = fft_size + cp_len
    rx_grid = np.zeros((num_subcarriers, num_symbols), dtype=complex)
    
    start_idx = (fft_size - num_subcarriers) // 2
    
    for i in range(num_symbols):
        start = i * symbol_len
        end = start + symbol_len
        symbol_with_cp = rx_signal[start:end]
        symbol_no_cp = symbol_with_cp[cp_len:]
        freq_data = np.fft.fftshift(np.fft.fft(symbol_no_cp))  # fftshift để khớp ifftshift
        rx_grid[:, i] = freq_data[start_idx:start_idx + num_subcarriers]

    return rx_grid
