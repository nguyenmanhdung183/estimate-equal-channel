from simulate import *
from fading import *
from gauss import *
import random
import numpy as np
from scipy import stats

def simulate_10000(number_of_loop, equalizer_mode, mod_type, X):
    '''
    Mô phỏng truyền dẫn OFDM qua kênh fading + AWGN và cân bằng bằng ZF hoặc MMSE.
    Lặp lại number_of_loop lần để tính BER và MSE trung bình theo từng mức SNR.
    
    Parameters:
        - number_of_loop: số lần lặp mô phỏng
        - equalizer_mode: 'ZF' hoặc 'MMSE'
        - mod_type: kiểu điều chế ('QPSK', '16QAM', '64QAM')
        - X: vector DMRS reference dùng cho ước lượng kênh
        
    Returns:
        - snr_range_db: danh sách các giá trị SNR đã chạy
        - ber_avg: danh sách BER trung bình ứng với mỗi SNR
        - mse_avg: danh sách MSE trung bình ứng với mỗi SNR
        - ber_ci: confidence intervals for BER values
        - mse_ci: confidence intervals for MSE values
    '''

    x_time = 14        # số cột (OFDM symbols)
    y_freq = 3276      # số subcarriers
    fft_size = 4096
    cp_len = 288
    snr_range_db = list(range(0, 32, 2))  # từ 0 đến 30 dB, bước 2
    fs = 30.72e6       # sample rate (Hz)

    # Xác định số bit trên mỗi symbol
    if mod_type == "QPSK":
        z_mod = 2
    elif mod_type == "16QAM":
        z_mod = 4
    elif mod_type == "64QAM":
        z_mod = 6
    else:
        raise ValueError("Modulation type not supported")

    # Arrays to store results for each iteration
    ber_results = np.zeros((len(snr_range_db), number_of_loop))
    mse_results = np.zeros((len(snr_range_db), number_of_loop))

    try:
        for num in range(number_of_loop):
            # 1. Sinh chuỗi bit ngẫu nhiên
            bits = ''.join(str(random.randint(0, 1)) for _ in range((x_time - 1) * y_freq * z_mod))
            symbol_mod = modulation(bits, mod_type)

            # 2. Ánh xạ symbol vào lưới tài nguyên
            grid = np.zeros((y_freq, x_time), dtype=complex)
            for i in range(y_freq):
                for j in range(x_time):
                    if j < 3:
                        grid[i, j] = symbol_mod[j * y_freq + i]
                    elif j == 3:
                        grid[i, j] = X[i]  # DMRS
                    else:
                        grid[i, j] = symbol_mod[(j - 1) * y_freq + i]

            # 3. OFDM modulation (bao gồm thêm CP)
            ofdm = np.zeros((fft_size + cp_len, x_time), dtype=complex)
            for i in range(x_time):
                ofdm[:, i] = ofdm_single_symbol(grid[:, i], fft_size, cp_len)

            # 4. Nối thành chuỗi OFDM dài và thêm padding
            ofdm_signal = ofdm.flatten(order='F')
            padding = np.zeros(64, dtype=complex)
            ofdm_signal = np.concatenate((ofdm_signal, padding))

            # 5. Kênh fading
            channel_h = generate_tdlb_channel(fs)
            faded_signal = apply_fading_channel(ofdm_signal, channel_h)

            # 6. Vòng lặp theo từng mức SNR
            for i, snr_db_test in enumerate(snr_range_db):
                try:
                    # Thêm AWGN
                    rx_signal = add_awgn(faded_signal, snr_db=snr_db_test)

                    # OFDM giải điều chế
                    rx_grid = ofdm_demodulate(rx_signal, fft_size=fft_size, cp_len=cp_len, num_symbols=x_time)
                    rx_grid = rx_grid[:y_freq, :]  # chỉ lấy 3276 subcarriers

                    # Ước lượng kênh từ DMRS (cột 3)
                    y_dmrs = rx_grid[:, 3]
                    if equalizer_mode == 'ZF':
                        h_est = estimate_channel_ls(X, y_dmrs)
                        rx_grid_eq = equalize_zf(rx_grid, h_est)
                    elif equalizer_mode == 'MMSE':
                        h_est = estimate_channel_mmse(X, y_dmrs, snr_db=snr_db_test)
                        rx_grid_eq = equalize_mmse(rx_grid, h_est, snr_db=snr_db_test)
                    else:
                        raise ValueError("Chỉ hỗ trợ 'ZF' hoặc 'MMSE'.")

                    # Giải điều chế các symbol thành bit (bỏ cột DMRS)
                    rx_bits = []
                    for col in [j for j in range(x_time) if j != 3]:
                        rx_symbols = rx_grid_eq[:, col]
                        rx_bits.extend(demodulation(rx_symbols, mod_type))

                    tx_bits_array = np.array([int(b) for b in bits], dtype=int)
                    rx_bits_array = np.array(rx_bits[:len(tx_bits_array)], dtype=int)

                    # Tính BER & MSE
                    ber_results[i, num] = np.mean(rx_bits_array != tx_bits_array)
                    mse_results[i, num] = np.mean(np.abs(rx_grid_eq - grid) ** 2)

                    print(f"Loop {num+1}/{number_of_loop} | SNR = {snr_db_test} dB → BER = {ber_results[i, num]:.5f}, MSE = {mse_results[i, num]:.5f}")

                except Exception as e:
                    print(f"Error in SNR loop {snr_db_test}dB: {str(e)}")
                    ber_results[i, num] = np.nan
                    mse_results[i, num] = np.nan

    except Exception as e:
        print(f"Error in main simulation loop: {str(e)}")
        return None, None, None, None, None

    # Calculate means and confidence intervals
    confidence_level = 0.95
    ber_avg = np.nanmean(ber_results, axis=1)
    mse_avg = np.nanmean(mse_results, axis=1)
    
    # Calculate confidence intervals
    ber_ci = np.zeros((len(snr_range_db), 2))
    mse_ci = np.zeros((len(snr_range_db), 2))
    
    for i in range(len(snr_range_db)):
        # Remove NaN values
        valid_ber = ber_results[i, ~np.isnan(ber_results[i, :])]
        valid_mse = mse_results[i, ~np.isnan(mse_results[i, :])]
        
        if len(valid_ber) > 1:  # Need at least 2 points for confidence interval
            ber_ci[i] = stats.t.interval(confidence_level, len(valid_ber)-1, 
                                       loc=np.mean(valid_ber), 
                                       scale=stats.sem(valid_ber))
        else:
            ber_ci[i] = [np.nan, np.nan]
            
        if len(valid_mse) > 1:
            mse_ci[i] = stats.t.interval(confidence_level, len(valid_mse)-1,
                                       loc=np.mean(valid_mse),
                                       scale=stats.sem(valid_mse))
        else:
            mse_ci[i] = [np.nan, np.nan]

    return snr_range_db, ber_avg.tolist(), mse_avg.tolist(), ber_ci.tolist(), mse_ci.tolist()
