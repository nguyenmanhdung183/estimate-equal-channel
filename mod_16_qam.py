import numpy as np

def bits_to_symbols_16QAM(bit_string):
    """
    Điều chế 16-QAM từ chuỗi bit.
    bit_string: chuỗi bit, ví dụ '0100110010...', độ dài phải chia hết cho 4.
    Trả về: mảng phức (complex) biểu diễn tín hiệu 16-QAM.
    """
    # Đảm bảo độ dài chuỗi bit chia hết cho 4
    if len(bit_string) % 4 != 0:
        raise ValueError("Độ dài chuỗi bit phải chia hết cho 4.")
    
    # Mapping bảng 16-QAM Gray code (4 bit -> 1 symbol)
    # Bảng tham khảo: mỗi 2 bit đầu quyết định I, 2 bit cuối quyết định Q
    # Mapping 2 bit thành giá trị mức biên độ (ví dụ: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3)
    mapping = {
        '00': -3,
        '01': -1,
        '11': 1,
        '10': 3
    }
    
    symbols = []
    for i in range(0, len(bit_string), 4):
        bits_i = bit_string[i:i+2]   # 2 bit cho I
        bits_q = bit_string[i+2:i+4] # 2 bit cho Q
        
        I = mapping[bits_i]
        Q = mapping[bits_q]
        
        # Tín hiệu phức
        symbol = complex(I, Q)
        symbols.append(symbol)
        
    # Chuẩn hóa năng lượng tín hiệu trung bình về 1
    symbols = np.array(symbols)
    symbols /= np.sqrt((np.mean(np.abs(symbols)**2)))
    
    return symbols





def qam16_demod_hard(symbols):
    bits_out = []
    for sym in symbols:
        I = np.real(sym) * np.sqrt(10)
        Q = np.imag(sym) * np.sqrt(10)

        # I bits
        if I < -2:
            bits_out.extend([0, 0])
        elif I < 0:
            bits_out.extend([0, 1])
        elif I < 2:
            bits_out.extend([1, 1])
        else:
            bits_out.extend([1, 0])

        # Q bits
        if Q < -2:
            bits_out.extend([0, 0])
        elif Q < 0:
            bits_out.extend([0, 1])
        elif Q < 2:
            bits_out.extend([1, 1])
        else:
            bits_out.extend([1, 0])
    return bits_out
