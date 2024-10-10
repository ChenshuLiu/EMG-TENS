import numpy as np

def period_calc(samples_per_second):
    # period is the time interval between two collections
    #sample_rate = 1/(period/1000)
    period = 1000/samples_per_second
    return period

class RollingBuffer:
    def __init__(self, size, dtype = np.float64) -> None:
        self.bufferstorage = np.zeros(size, dtype=dtype)
        self.size = size
        self.current_size = 0
    def add(self, element):
        if self.current_size < self.size:
            self.bufferstorage[self.current_size] = element
            self.current_size += 1
        else:
            self.bufferstorage[:-1] = self.bufferstorage[1:] # shifting to the left (dropping the first element)
            self.bufferstorage[-1] = element
    def get(self):
        return self.bufferstorage[:self.current_size]