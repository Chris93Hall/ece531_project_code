"""
main.py
"""

import numpy as np

import pluto
import frame_generator

def sim():
    frame = list(frame_generator.get_frame())
    frame += frame
    frame_rx = np.convolve(frame, [1, 0, 0.25], mode='same')
    frame_generator.process_frame(frame_rx, frame)


def main():
    frame = list(frame_generator.get_frame())
    frame += frame
    frame_rx = np.convolve(frame, [1, 0, 0.25], mode='same')
    frame_rx = pluto.transmit(frame)
    frame_generator.process_frame(frame_rx, frame)

if __name__ == '__main__':
    main()
