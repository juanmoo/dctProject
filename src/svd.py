import numpy as np
from numpy.linalg import svd
from utils import *
from math import prod
from common import Encoding

class SVD(Encoding):

    # __init__ inherited from Encoding

    def encode(self, block_size=8, frac=1.0):

        self.block_size = block_size
        self.frac = frac

        self.rnum, self.cnum = self.img.shape[:-1]
        self.rpad = (self.block_size - self.rnum%self.block_size)%self.block_size
        self.cpad = (self.block_size - self.cnum%self.block_size)%self.block_size

        # Transform Coding
        self.blocks = []

        for channel in range(self.img.shape[-1]):
            channel_blocks = []

            padded = np.pad(self.img[:, :, channel], ((0, self.rpad), (0, self.cpad)), 'constant')
            rnum, cnum = padded.shape

            for i in range(rnum//self.block_size):
                for j in range(cnum//self.block_size):
                    block = padded[block_size * i:block_size * (i + 1), block_size * j: block_size * (j + 1)]
                    svd = svd_factor(block, frac=frac)
                    channel_blocks.append(svd)
        
            self.blocks.append(channel_blocks)

        return self

    def calculate_size(self):
        return None

    def decode(self):
        output = np.zeros(self.img.shape)

        cnum = self.img.shape[1] + self.cpad
        horizontal_block_count = cnum // self.block_size

        for channel in range(self.img.shape[2]):
            for index, (u, v, d, mean) in enumerate(self.blocks[channel]):
                i = index // horizontal_block_count
                j = index % horizontal_block_count

                r_min = i * self.block_size
                r_max = min((i + 1) * self.block_size, self.img.shape[0])
                row_diff = r_max - r_min

                c_min = j * self.block_size
                c_max = min((j + 1) * self.block_size, self.img.shape[1])
                col_diff = c_max - c_min

                block = (u @ np.diag(d) @ v.T) + mean

                output[r_min:r_max, c_min:c_max, channel] = block[:row_diff, :col_diff]
        
        return output
    
def svd_factor(arr, frac=1.0):
    # Rank 1 Update
    mean = arr.mean()
    u, sigma, vh = svd(arr - mean)

    if frac < 1.0:
        rank = 0
        tot = 0.0
        max_sum = frac * sigma.sum()

        for sv in sigma:
            if (max_sum >= tot + sv):
                tot += sv
                rank += 1
            else:
                break
    else:
        rank = len(sigma)

    u = u[:, :rank]
    sigma = sigma[:rank]
    vh = vh[:rank, :]

    return (u, vh.T, sigma, mean)


if __name__ == '__main__':
    arr = np.random.randint(100, size=(10, 10, 3))

    print('arr:', arr.shape)

    factorization = SVD(arr)
    factorization.encode(block_size=8, frac=1.0)

    print(len(factorization.blocks[0]))
    new_arr = factorization.decode()

    print('decoded:', new_arr)


    err = (arr - new_arr)
    err = err * err
    err = err.sum()

    print('Error:', err)
