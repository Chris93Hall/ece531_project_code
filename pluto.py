"""
pluto.py
"""

import adi

def transmit(frame):
    sdr = adi.Pluto()
    sdr.rx_rf_bandwidth = 4000000
    sdr.rx_lo = 2000000000
    sdr.tx_lo = 2000000000
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = 0
    sdr.gain_control_mode_chan0 = 'slow_attack'
    sdr.rx_buffer_size = 32764

    sdr.tx(frame)
    rx_frame = sdr.rx()
    return rx_frame
