"""
phy_frame.py

Contains class FrameGenerator which can be used to
build physical layer frames for experiments and
simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

class PhyFrame():
    def __init__(self,frame=None, length=16384):
        if not frame is None:
            self.frame = np.array(frame).astype(complex)
            self.length = len(self.frame)
        else:
            self.length = length
            self.frame = np.array([0j]*self.length)
        self.block_start_indices = []
        self.block_lengths = []
        self.block_labels = [] 
        return

    def __len__(self):
        return len(self.frame)

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as fd:
            pickle.dump(self, fd)
        return

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as fd:
            obj = pickle.load(fd)
        
        # These exists a better way to iterate over these
        self.length = obj.length
        self.frame = np.array(obj.frame)
        self.block_start_indices = obj.block_start_indices
        self.block_lengths = obj.block_lengths
        self.block_labels = obj.block_labels
        return

    def set_length(self, length):
        self.length = length
        if length < len(self.frame):
            self.frame = self.frame[0:length]
        elif length > len(self.frame):
            len_diff = length - len(self.frame)
            append_list = [0j]*len_diff
            self.frame = self.frame + append_list
        return

    def set_block_start_indices(self, index_list):
        self.block_start_indices = index_list
        self.block_lengths = [0]*len(index_list)
        self.block_labels = ['']*len(index_list)
        return

    def get_block(self, block_index):
        start_index = self.block_start_indices[block_index]
        block_length = self.block_lengths[block_index]
        return self.frame[start_index:start_index + block_length] 

    def get_blocks(self):
        blocks_list = []
        for index in range(len(self.block_start_indices)):
            start_index = self.block_start_indices[index]
            block_length = self.block_lengths[index]
            blocks_list.append(self.frame[start_index:start_index + block_length]) 
        return blocks_list

    def set_block(self, index, vector):
        start_index = self.block_start_indices[index]
        length = len(vector)
        self.block_lengths[index] = length
        self.frame[start_index:start_index+length] = vector
        return

    def set_blocks(self, vector_list):
        for index, vector in enumerate(vector_list):
            self.set_block(index, vector)
        return

    def plot_frame(self):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(np.abs(self.frame))
        plt.subplot(3,1,2)
        plt.plot(np.real(self.frame))
        plt.subplot(3,1,3)
        plt.plot(np.imag(self.frame))
        plt.show()
        return
        
