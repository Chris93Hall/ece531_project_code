"""
frame_gen.py
"""

import numpy as np
import matplotlib.pyplot as plt

from phy_frame import PhyFrame
import signal_generator

def get_frame():
    frame = PhyFrame(length=16382)

    frame.set_block_start_indices([256, 1024, 6143, 8191, 10239, 12287])

    bpsk = signal_generator.BPSK(256, 2)

    #chirp1 = signal_generator.complex_chirp(512, 0, np.pi)
    #chirp2 = signal_generator.complex_chirp(512, np.pi, 0)
    #chirp = np.hstack((chirp2[-64:], chirp1, chirp2, chirp1, chirp2, chirp1, chirp2, chirp1, chirp2, chirp1[0:64]))

    chirp1 = signal_generator.complex_chirp2(1024, 0, np.pi)
    chirp = np.hstack((chirp1[-64:], chirp1, chirp1, chirp1, chirp1, chirp1[0:64]))

    ofdm1 = signal_generator.OFDM_4QAM(1024, cpf_length=64)
    ofdm2 = signal_generator.OFDM_4QAM(1024, cpf_length=64)
    ofdm3 = signal_generator.OFDM_4QAM(1024, cpf_length=64)
    ofdm4 = signal_generator.OFDM_4QAM(1024, cpf_length=64)

    frame.set_block(0, bpsk)
    frame.set_block(1, chirp)
    frame.set_block(2, ofdm1)
    frame.set_block(3, ofdm2)
    frame.set_block(4, ofdm3)
    frame.set_block(5, ofdm4)

    #frame.plot_frame()
    return frame.frame

def process_frame(frame_rx, frame_tx):
    frame_tx = PhyFrame(frame_tx)
    frame_tx.set_block_start_indices([256, 1024, 6143, 8191, 10239, 12287]) 
    frame_tx.block_lengths = [512, 4224, 1152, 1152, 1152, 1152]

    plt.figure()
    plt.plot(np.abs(frame_rx))
    plt.show(block=False)

    # Frame sync
    bpsk = frame_tx.get_block(0)
    bpsk_diff = np.sign(bpsk[2::2] - bpsk[0:-2:2])
 
    frame_tx_diff = np.sign(frame_rx[2::2] - frame_rx[0:-2:2])
    start_index = np.argmax(np.abs(np.convolve(frame_tx_diff, np.flip(bpsk_diff), mode='same'))) + 1
    print(start_index)

    # Correlation
    plt.figure()
    plt.plot(np.abs(np.convolve(frame_tx_diff, np.flip(bpsk_diff))))
    plt.grid(True)
    plt.show(block=False)

    start_index -= 256

    # Chirp
    chirp = frame_rx[start_index+1024+64:start_index+1024+64+(4*1024)]
    chirp1 = np.fft.fft(chirp[0:1024])
    chirp2 = np.fft.fft(chirp[1024:2048])
    chirp3 = np.fft.fft(chirp[2048:3072])
    chirp4 = np.fft.fft(chirp[3072:4096])

    #plt.figure()
    #plt.plot(np.real(chirp))
    #plt.show(block=False)

    # OFDM
    ofdm1 = np.fft.fft(frame_rx[start_index+6143:start_index+6143+1024])
    ofdm2 = np.fft.fft(frame_rx[start_index+8191:start_index+8191+1024])
    ofdm3 = np.fft.fft(frame_rx[start_index+10239:start_index+10239+1024])
    ofdm4 = np.fft.fft(frame_rx[start_index+12287:start_index+12287+1024])

    ofdm1_tx = np.fft.fft(frame_tx.get_block(2)[64:-64])
    ofdm2_tx = np.fft.fft(frame_tx.get_block(3)[64:-64])
    ofdm3_tx = np.fft.fft(frame_tx.get_block(4)[64:-64])
    ofdm4_tx = np.fft.fft(frame_tx.get_block(5)[64:-64])

    if False:
        plt.figure()
        plt.subplot(8,1,1)
        plt.plot(np.abs(chirp1))
        plt.subplot(8,1,2)
        plt.plot(np.angle(chirp1))
        plt.subplot(8,1,3)
        plt.plot(np.abs(chirp2))
        plt.subplot(8,1,4)
        plt.plot(np.angle(chirp2))
        plt.subplot(8,1,5)
        plt.plot(np.abs(chirp3))
        plt.subplot(8,1,6)
        plt.plot(np.angle(chirp3))
        plt.subplot(8,1,7)
        plt.plot(np.abs(chirp4))
        plt.subplot(8,1,8)
        plt.plot(np.angle(chirp4))
        plt.show(block=False)
        
    if False:
        plt.figure()
        plt.subplot(8,1,1)
        plt.plot(np.abs(ofdm1))
        plt.subplot(8,1,2)
        plt.plot(np.angle(ofdm1))
        plt.subplot(8,1,3)
        plt.plot(np.abs(ofdm2))
        plt.subplot(8,1,4)
        plt.plot(np.angle(ofdm2))
        plt.subplot(8,1,5)
        plt.plot(np.abs(ofdm3))
        plt.subplot(8,1,6)
        plt.plot(np.angle(ofdm3))
        plt.subplot(8,1,7)
        plt.plot(np.abs(ofdm4))
        plt.subplot(8,1,8)
        plt.plot(np.angle(ofdm4))
        plt.show(block=True)

    plt.figure()
    plt.subplot(8,1,1)
    plt.plot(np.abs(ofdm1/ofdm1_tx))
    plt.subplot(8,1,2)
    plt.plot(np.angle(ofdm1/ofdm1_tx))
    plt.subplot(8,1,3)
    plt.plot(np.abs(ofdm2/ofdm2_tx))
    plt.subplot(8,1,4)
    plt.plot(np.angle(ofdm2/ofdm2_tx))
    plt.subplot(8,1,5)
    plt.plot(np.abs(ofdm3/ofdm3_tx))
    plt.subplot(8,1,6)
    plt.plot(np.angle(ofdm3/ofdm3_tx))
    plt.subplot(8,1,7)
    plt.plot(np.abs(ofdm4/ofdm4_tx))
    plt.subplot(8,1,8)
    plt.plot(np.angle(ofdm4/ofdm4_tx))
    plt.show(block=False)

    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(np.abs(ofdm1/ofdm1_tx))
    plt.subplot(4,1,2)
    plt.plot(np.abs(ofdm2/ofdm2_tx))
    plt.subplot(4,1,3)
    plt.plot(np.abs(ofdm3/ofdm3_tx))
    plt.subplot(4,1,4)
    plt.plot(np.abs(ofdm4/ofdm4_tx))
    plt.show(block=True)
    
    return

if __name__ == '__main__':
    frame_rx = get_frame()
    process_frame(frame_rx, frame_rx)
