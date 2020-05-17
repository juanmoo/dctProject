import numpy as np
from math import cos, sqrt, pi
from common import Encoding

class DCT(Encoding):

    # __init__ inherited from Encoding

    def encode(self, block_size=8, mult=1.0):

        self.block_size = block_size
        self.mult = mult

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
                    coeffs = dct_factor(block)
                    channel_blocks.append(coeffs)
        
            self.blocks.append(channel_blocks)

        return self

    def decode(self):
        output = np.zeros(self.img.shape)

        cnum = self.img.shape[1] + self.cpad
        horizontal_block_count = cnum // self.block_size

        for channel in range(self.img.shape[2]):
            for index, coeffs in enumerate(self.blocks[channel]):
                i = index // horizontal_block_count
                j = index % horizontal_block_count

                r_min = i * self.block_size
                r_max = min((i + 1) * self.block_size, self.img.shape[0])
                row_diff = r_max - r_min

                c_min = j * self.block_size
                c_max = min((j + 1) * self.block_size, self.img.shape[1])
                col_diff = c_max - c_min

                block = inv_dct(coeffs)

                output[r_min:r_max, c_min:c_max, channel] = block[:row_diff, :col_diff]
        
        return output

    def calculate_size(self):
        pass


memo = dict()
def make_dct_basis(dim):
    if dim in memo:
        return memo[dim]

    basis = np.zeros((dim, dim))
    alpha = lambda i, j: sqrt(1.0/dim) if i == 0 else sqrt(2.0/dim)

    for i in range(dim):
        for j in range(dim):
            basis[i, j] = alpha(i, j) * cos(pi * (2 * j + 1) * i/(2.0 * dim))

    memo[dim] = basis

    return basis

def dct_factor(arr):
    m, n = arr.shape[:2]
    assert(m == n)
    basis = make_dct_basis(m)
    return np.round(basis.T@(arr - 128.0)@basis)

def inv_dct(coeffs):
    m, n = coeffs.shape[:2]
    assert(m == n)
    basis = make_dct_basis(m)
    return (basis@coeffs@basis.T) + 128.0



if __name__ == '__main__':
    d = 30

    arr = np.random.randint(10, size=(d,d))

    dct_C = dct_factor(arr)
    decoded = inv_dct(dct_C)
    # decoded = np.array(decoded, dtype=np.uint8)
    print((arr - decoded).sum()/d**2)