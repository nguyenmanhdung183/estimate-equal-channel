
from gauss import *
import numpy as np
from ofdm import *
from estimate import *
from equalize import *
from mod import *


def simulate(faded_signal, X, bits, grid, equalizer_mode, mod_type): #thay đổi SNR
    snr_range_db = list(range(0, 32, 2))  # 0 → 30 dB
    ber_results = []
    mse_results = []


    for snr_db_test in snr_range_db:

        # 1. Thêm AWGN vào tín hiệu OFDM sau fading
        rx_signal_test = add_awgn(faded_signal, snr_db=snr_db_test)

        # 2. Giải điều chế OFDM
        rx_grid_test = ofdm_demodulate(rx_signal_test, fft_size=4096, cp_len=288, num_symbols=14)
        rx_grid_test = rx_grid_test[:3276, :]

        # 3. Ước lượng kênh từ cột DMRS (col 3)
        dmrs_col = 3
        y_dmrs = rx_grid_test[:, dmrs_col]

        if equalizer_mode == 'ZF':
            h_estimated = estimate_channel_ls(X, y_dmrs)
        elif equalizer_mode == 'MMSE':
            h_estimated = estimate_channel_mmse(X, y_dmrs, snr_db=snr_db_test)
        else:
            raise ValueError("Chỉ hỗ trợ 'ZF' hoặc 'MMSE'.")

        # 4. Cân bằng tín hiệu

        if equalizer_mode == 'ZF':
            rx_grid_eq = equalize_zf(rx_grid_test, h_estimated)
        else:  # MMSE
            rx_grid_eq = equalize_mmse(rx_grid_test, h_estimated, snr_db=snr_db_test)

        # 5. Giải điều chế các symbol thành bit
        data_cols_test = [i for i in range(14) if i != 3]  # loại bỏ cột 3 (DMRS)
        rx_bits_full = []
        for col in data_cols_test:
            rx_symbols_test = rx_grid_eq[:, col]
            bits_col_test = demodulation(rx_symbols_test, mod_type)
            rx_bits_full.extend(bits_col_test)

        tx_bits_array = np.array([int(b) for b in bits], dtype=int)
        rx_bits_array = np.array(rx_bits_full[:len(tx_bits_array)], dtype=int)  # Cắt cho khớp

        # 6. Tính toán BER và MSE
        ber_val = np.mean(rx_bits_array != tx_bits_array)
        mse_val = np.mean(np.abs(rx_grid_eq - grid)**2)

        print(f"SNR = {snr_db_test} dB → BER = {ber_val:.5f}, MSE = {mse_val:.5f}")

        ber_results.append(ber_val)
        mse_results.append(mse_val)
        
    return snr_range_db, ber_results, mse_results