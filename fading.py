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



def generate_rayleigh_channel(sampling_rate_hz):
    """
    Tạo kênh Rayleigh đơn giản với một tap duy nhất.
    Mô phỏng kênh fading với phân phối Rayleigh cho biên độ và pha ngẫu nhiên.
    
    Tham số:
    - sampling_rate_hz: Tần số mẫu (Hz), dùng để tính toán độ trễ nếu cần thiết.

    Trả về:
    - h: Phản hồi kênh (tap duy nhất) dưới dạng mảng phức.
    """
    # Mô phỏng fading Rayleigh với biên độ theo phân phối Rayleigh và pha ngẫu nhiên.
    # Biên độ Rayleigh
    amplitude = np.sqrt(1 / 2) * (np.random.randn() + 1j * np.random.randn())
    
    # Kênh có pha ngẫu nhiên
    h = amplitude
    
    # Chuẩn hóa kênh
    h /= np.linalg.norm(h)

    return h


def apply_rayleigh_channel(signal, channel_response):
    """
    Áp dụng kênh Rayleigh lên tín hiệu.
    
    Tham số:
    - signal: Tín hiệu đầu vào, có thể là OFDM symbol hay tín hiệu đơn giản.
    - channel_response: Phản hồi kênh (tap kênh) dưới dạng phức.
    
    Trả về:
    - tín hiệu đã bị fade.
    """
    # Áp dụng kênh lên tín hiệu
    faded_signal = signal * channel_response  # Cộng hưởng tín hiệu với kênh
    
    return faded_signal