"""
signal_generator.py

TODO: Docstrings
      redefine complex (warning)
"""

import numpy as np

def QAM4(shape):
    real_symbols = np.random.randint(0, 2, shape)*2-1
    imag_symbols = np.random.randint(0, 2, shape)*2-1

    time_domain_signal = real_symbols + 1j*imag_symbols
    return time_domain_signal

def QAM16(shape):
    real_symbols = np.random.randint(0, 4, shape)-1.5
    imag_symbols = np.random.randint(0, 4, shape)-1.5

    time_domain_signal = real_symbols + 1j*imag_symbols
    time_domain_signal = time_domain_signal/1.5
    return time_domain_signal

def BPSK(shape, symb_dur=1):
    real_symbols = np.random.randint(0,2,shape)*2-1
    if symb_dur == 1:
        return real_symbols
    real_samples = np.repeat(real_symbols, symb_dur)
    return real_samples

def PSK(length, constellation_size):
    symbols = np.random.randint(0, constellation_size, (length,))
    angle_delta = 2.0*np.pi/constellation_size
    symbol_angles = symbols*angle_delta
    time_domain_signal = np.cos(symbol_angles) + 1j*np.sin(symbol_angles)
    return time_domain_signal

def APSK16(shape):
    symbols = list(np.random.randint(0, 16, shape))
    signal = [0]*length
    for index in range(len(signal)):
        symbol = symbols[index]    
        if symbol < 4:
            angle_delta = 2.0*np.pi/4
            symbol_angle = symbol*angle_delta
            signal[index] = np.cos(symbol_angle) + 1j*np.sin(symbol_angle)
        elif symbol >= 4:
            symbol = symbol - 4
            angle_delta = 2.0*np.pi/12
            symbol_angle = symbol*angle_delta
            signal[index] = 2*np.cos(symbol_angle) + 2j*np.sin(symbol_angle)
    return signal

def arbitrary_QAM(shape, symbol_coord_list):
    """
    symbol_coord_list should be a list of tuples.
    """
    num_symbols = len(symbol_coord_list)
    symbols = list(np.random.randint(0, num_symbols, shape))
    signal = []
    for index in range(len(symbols)):
        coord_temp = symbol_coord_list(symbols[index])
        signal.append(coord_temp[0] + 1j*coord_temp[1])
    return signal

def OFDM_4QAM(shape, cpf_length=0, bitmap=None):
    real_subcarriers = np.random.randint(0, 2, shape)*2-1
    imag_subcarriers = np.random.randint(0, 2, shape)*2-1

    freq_domain_signal = real_subcarriers + 1j*imag_subcarriers

    if bitmap:
        freq_domain_signal = freq_domain_signal*np.array(bitmap)

    # inverse fourier transform
    time_domain_signal = np.fft.ifft(freq_domain_signal)

    # add cyclic prefix and postfix
    postfix = time_domain_signal[0:cpf_length]
    prefix = time_domain_signal[-cpf_length:]
    time_domain_signal = np.hstack((prefix, time_domain_signal, postfix))
    time_domain_signal /= np.max(np.abs(time_domain_signal))
    return time_domain_signal

def OFDM_16QAM(shape, cpf_length=0):
    real_subcarriers = np.random.randint(0, 4, shape)-1.5
    imag_subcarriers = np.random.randint(0, 4, shape)-1.5

    freq_domain_signal = real_subcarriers + 1j*imag_subcarriers

    # inverse fourier transform
    time_domain_signal = np.fft.ifft(freq_domain_signal)

    # add cyclic prefix and postfix
    postfix = time_domain_signal[0:cpf_length]
    prefix = time_domain_signal[-cpf_length:]
    time_domain_signal = np.hstack((prefix, time_domain_signal, postfix))
    time_domain_signal /= np.max(np.abs(time_domain_signal))
    return time_domain_signal

def gaussian_noise(shape, variance=1.0, is_complex=False):
    noise_sig = np.random.normal(0.0, np.sqrt(variance), shape)
    if is_complex is True:
        noise_re = (noise_sig/np.sqrt(2.0))
        noise_im = np.random.normal(0.0, np.sqrt(variance/2.0), shape)
        noise_re = noise_re.astype(complex)
        noise_im = noise_im.astype(complex)
        noise_sig = noise_re + 1j*noise_im
    return noise_sig

def sinusoid(length, freq):
    """
    freq given in radians per sample
    """
    index = np.array(range(length))
    sig = np.cos(freq*index)
    return sig

def chirp(length, start_freq, end_freq, start_phase=0):
    index = np.array(range(length))
    freq_slope = (end_freq-start_freq)/float(length)
    chirp_sig = np.cos(0.5*freq_slope*(index**2) + start_freq*index + start_phase)
    return chirp_sig

def complex_chirp(length, start_freq, end_freq, start_phase=0):
    index = np.array(range(length))
    freq_slope = (end_freq-start_freq)/float(length)
    chirp_sig = np.exp(1j*(0.5*freq_slope*(index**2) + start_freq*index + start_phase))
    return chirp_sig

def complex_chirp2(length, start_freq, end_freq, start_phase=0):
    index = np.array(range(length)) - length/2.0
    freq_slope = (end_freq-start_freq)/float(length)
    chirp_sig = np.exp(1j*(2.0*np.pi/length)*index**2)
    return chirp_sig

def square_wave(length, on_length, off_length):
    sig = [0.0]*length
    period = on_length + off_length
    for index in range(length):
        if index%period < on_length:
            sig[index] = 1.0
    return np.array(sig)

def test_signal1(length):
    freq = np.zeros(length)
    index_off_center = length//16
    if length % 2 == 1:
        center_index = int((length/2.0) - 0.5)
        freq[center_index-index_off_center:center_index] = 1
        freq[center_index:center_index+index_off_center-1] = 1.0 - (np.arange(1, index_off_center)/index_off_center)
        freq = np.fft.ifftshift(freq)
        time = np.fft.ifft(freq)
    else:
        center_index = int(length/2.0) 
        center_index = [center_index-1, center_index]
        freq[center_index[0]-index_off_center: center_index[1]] = 1
        freq[center_index[1]:center_index[1]+index_off_center-1] = 1.0 - (np.arange(1, index_off_center)/index_off_center)
        freq = np.fft.ifftshift(freq)
        time = np.fft.ifft(freq)
    return time

