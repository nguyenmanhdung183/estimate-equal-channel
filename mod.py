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


def qpsk_mod(bit_string):
    """
    Điều chế QPSK từ chuỗi bit.
    bit_string: chuỗi bit, độ dài phải chia hết cho 2.
    Trả về: mảng complex biểu diễn tín hiệu QPSK.
    """
    if len(bit_string) % 2 != 0:
        raise ValueError("Độ dài chuỗi bit phải chia hết cho 2.")
    
    # Mapping 2-bit thành điểm QPSK theo Gray coding
    mapping = {
        '00': complex(1, 1),
        '01': complex(-1, 1),
        '11': complex(-1, -1),
        '10': complex(1, -1)
    }

    symbols = []
    for i in range(0, len(bit_string), 2):
        bits = bit_string[i:i+2]
        symbols.append(mapping[bits])

    symbols = np.array(symbols)
    symbols /= np.sqrt(np.mean(np.abs(symbols)**2))  # Chuẩn hóa năng lượng trung bình về 1
    return symbols

def qpsk_demod_hard(symbols):
    """
    Giải điều chế cứng QPSK (hard decision).
    symbols: mảng complex QPSK đã nhận.
    Trả về: danh sách bit.
    """
    bits_out = []
    for sym in symbols:
        I = np.real(sym) * np.sqrt(2)
        Q = np.imag(sym) * np.sqrt(2)

        if I > 0 and Q > 0:
            bits_out += [0, 0]
        elif I < 0 and Q > 0:
            bits_out += [0, 1]
        elif I < 0 and Q < 0:
            bits_out += [1, 1]
        else:  # I > 0 and Q < 0
            bits_out += [1, 0]
    return bits_out


def qam64_mod(bit_string):
    """
    Điều chế 64-QAM từ chuỗi bit.
    bit_string: chuỗi bit, ví dụ '0100110010...', độ dài phải chia hết cho 6.
    Trả về: mảng phức (complex) biểu diễn tín hiệu 64-QAM.
    """
    # Đảm bảo độ dài chuỗi bit chia hết cho 6
    if len(bit_string) % 6 != 0:
        raise ValueError("Độ dài chuỗi bit phải chia hết cho 6.")
    
    # Mapping bảng 64-QAM Gray code (6 bit -> 1 symbol)
    mapping = {
        '000': -7, '001': -5, '011': -3, '010': -1,
        '110': 1, '111': 3, '101': 5, '100': 7
    }
    
    symbols = []
    for i in range(0, len(bit_string), 6):
        bits_i = bit_string[i:i+3]   # 3 bit cho I
        bits_q = bit_string[i+3:i+6] # 3 bit cho Q
        
        I = mapping[bits_i]
        Q = mapping[bits_q]
        
        # Tín hiệu phức
        symbol = complex(I, Q)
        symbols.append(symbol)
        
    # Chuẩn hóa năng lượng tín hiệu trung bình về 1
    symbols = np.array(symbols)
    symbols /= np.sqrt((np.mean(np.abs(symbols)**2)))
    
    return symbols

def qam64_demod_hard(symbols):
    bits_out = []
    for sym in symbols:
        I = np.real(sym) * np.sqrt(42)
        Q = np.imag(sym) * np.sqrt(42)

        # I bits
        if I < -5:
            bits_out.extend([0, 0, 0])
        elif I < -3:
            bits_out.extend([0, 0, 1])
        elif I < -1:
            bits_out.extend([0, 1, 1])
        elif I < 1:
            bits_out.extend([0, 1, 0])
        elif I < 3:
            bits_out.extend([1, 1, 0])
        elif I < 5:
            bits_out.extend([1, 1, 1])
        else:
            bits_out.extend([1, 0, 1])

        # Q bits
        if Q < -5:
            bits_out.extend([0, 0, 0])
        elif Q < -3:
            bits_out.extend([0, 0, 1])
        elif Q < -1:
            bits_out.extend([0, 1, 1])
        elif Q < 1:
            bits_out.extend([0, 1, 0])
        elif Q < 3:
            bits_out.extend([1, 1, 0])
        elif Q < 5:
            bits_out.extend([1, 1, 1])
        else:
            bits_out.extend([1, 0, 1])
    
    return bits_out


def modulation(bit_string, modulation_type='16QAM'):
    """
    Điều chế chuỗi bit theo loại điều chế yêu cầu.
    bit_string: chuỗi bit, ví dụ '0100110010...'.
    modulation_type: '16QAM', 'QPSK', hoặc '64QAM'.
    Trả về: mảng phức (complex) biểu diễn tín hiệu đã điều chế.
    """
    if modulation_type == '16QAM':
        return bits_to_symbols_16QAM(bit_string)
    elif modulation_type == 'QPSK':
        return qpsk_mod(bit_string)
    elif modulation_type == '64QAM':
        return qam64_mod(bit_string)
    else:
        raise ValueError("Chỉ hỗ trợ các loại điều chế: '16QAM', 'QPSK', '64QAM'.")
    
def demodulation(symbols, modulation_type='16QAM'):
    """
    Giải điều chế tín hiệu theo loại điều chế yêu cầu.
    symbols: mảng phức (complex) biểu diễn tín hiệu đã điều chế.
    modulation_type: '16QAM', 'QPSK', hoặc '64QAM'.
    Trả về: chuỗi bit đã giải điều chế.
    """
    if modulation_type == '16QAM':
        return qam16_demod_hard(symbols)
    elif modulation_type == 'QPSK':
        return qpsk_demod_hard(symbols)
    elif modulation_type == '64QAM':
        return qam64_demod_hard(symbols)
    else:
        raise ValueError("Chỉ hỗ trợ các loại điều chế: '16QAM', 'QPSK', '64QAM'.")