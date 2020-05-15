import numpy as np
from numpy.linalg import svd
from utils import *
from math import prod

class IMG_SVD:
    # Class Methods
    def from_array(arr, block_size=8, frac=1.0, max_sum=None):
        r, c = arr.shape
        rpad = (block_size - r%block_size)%block_size
        cpad = (block_size - c%block_size)%block_size

        # Make dimensions of arr be multiples of block_size
        new_arr = np.pad(arr, ((0, rpad), (0, cpad)), 'constant')
        r, c = new_arr.shape

        blocks = []

        for i in range(r//block_size):
            for j in range(c//block_size):

                block = new_arr[block_size * i:block_size * (i + 1), block_size * j: block_size * (j + 1)]
                svd = svd_factor(block, frac=frac, max_sum=max_sum)
                blocks.append(svd)
        
        return new_arr, blocks
        

    # Instance Methods
    def __init__(self, arr, block_size=8, frac=1.0, max_sum=None):

        self.block_size = block_size
        self.original = arr
        self.padded, self.blocks = IMG_SVD.from_array(self.original, block_size=block_size, frac=frac, max_sum=max_sum)

    def recreate(self):
        output = np.zeros(self.original.shape)

        horizontal_block_count = self.padded.shape[1] // self.block_size

        for index, (u, v, d, mean) in enumerate(self.blocks):
            i = index // horizontal_block_count
            j = index % horizontal_block_count

            r_min = i * self.block_size
            r_max = min((i + 1) * self.block_size, self.original.shape[0])
            row_diff = r_max - r_min

            c_min = j * self.block_size
            c_max = min((j + 1) * self.block_size, self.original.shape[1])
            col_diff = c_max - c_min

            block = (u @ np.diag(d) @ v.T) + mean

            output[r_min:r_max, c_min:c_max] = block[:row_diff, :col_diff]
        
        return output

    
    def calculate_size(self):
        tot_size = 0.0
        for block in self.blocks:
            u, d, v, mean = block
            tot_size += prod(u.shape)
            tot_size += prod(v.shape)
            tot_size += prod(d.shape)
            tot_size += 1

        # return result in bytes where each float uses 3 bytes
        return tot_size * 3

    
def svd_factor(arr, frac=1.0, max_sum=None):
    mean = arr.mean()
    u, sigma, vh = svd(arr - mean)

    if frac < 1.0:
        rank = int(len(sigma) * frac * 10.0 + 0.5)//10

    elif max_sum is not None:
        rank = 0
        tot = 0.0

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

    return u, vh.T, sigma, mean


if __name__ == '__main__':
    arr = np.random.randint(100, size=(5, 8))

    print('arr:', arr)

    factorization = IMG_SVD(arr, block_size=8, frac=0.6)
    new_arr = factorization.recreate()

    print('decoded:', new_arr)


    err = (arr - new_arr)
    err = err * err
    err = err.sum()

    print('Error:', err)