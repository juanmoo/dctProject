import numpy as np
from math import cos, sqrt, pi
from common import Encoding
from utils import diagonalOrder, fillDiagonal
from huffman import HuffmanCoding

class DCT(Encoding):

    # __init__ inherited from Encoding

    def encode(self, block_size=8, mult=1.0):

        self.block_size = block_size
        self.mult = mult

        self.rnum, self.cnum = self.img.shape[:-1]
        self.rpad = (self.block_size - self.rnum%self.block_size)%self.block_size
        self.cpad = (self.block_size - self.cnum%self.block_size)%self.block_size

        # Transform Coding
        self.encoded_blocks = []

        for channel in range(self.img.shape[-1]):
            encoded_channel_blocks = []

            padded = np.pad(self.img[:, :, channel], ((0, self.rpad), (0, self.cpad)), 'constant')
            rnum, cnum = padded.shape

            for i in range(rnum//self.block_size):
                for j in range(cnum//self.block_size):
                    block = padded[block_size * i:block_size * (i + 1), block_size * j: block_size * (j + 1)]
                    is_Y = (channel == 0)
                    coeffs = dct_factor(block, mult=mult, is_Y=is_Y)
                    encoded_channel_blocks.append(self.encode_block(coeffs))
        
            self.encoded_blocks.append(encoded_channel_blocks)

        return self

    def encode_block(self, block):
        symbols = diagonalOrder(block)
        h = HuffmanCoding(symbols)
        encoded = h.compress()
        return (encoded, h.reverse_mapping)
        
    def decode(self):
        output = np.zeros(self.img.shape)

        cnum = self.img.shape[1] + self.cpad
        horizontal_block_count = cnum // self.block_size

        h_tmp = HuffmanCoding([])

        for channel in range(self.img.shape[2]):
            for index, (encoded, rev_map) in enumerate(self.encoded_blocks[channel]):

                i = index // horizontal_block_count
                j = index % horizontal_block_count

                r_min = i * self.block_size
                r_max = min((i + 1) * self.block_size, self.img.shape[0])
                row_diff = r_max - r_min

                c_min = j * self.block_size
                c_max = min((j + 1) * self.block_size, self.img.shape[1])
                col_diff = c_max - c_min

                h_tmp.reverse_mapping = rev_map
                zigzag = h_tmp.decompress(encoded)
                coeffs = fillDiagonal(zigzag, self.block_size)
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

Qy = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

Qc = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])

def dct_factor(arr, mult=1.0, is_Y=True):
    m, n = arr.shape[:2]
    assert(m == n)
    basis = make_dct_basis(m)
    coeffs = np.round(basis.T@(arr - 128.0)@basis)
    Q = Qy if is_Y else Qc
    coeffs = (1/Q) * coeffs
    return np.round(coeffs)

def inv_dct(coeffs, mult=1.0, is_Y=True):
    m, n = coeffs.shape[:2]
    assert(m == n)
    basis = make_dct_basis(m)
    Q = Qy if is_Y else Qc
    coeffs = (basis@coeffs@basis.T) + 128.0
    coeffs = Q * coeffs

    return coeffs




if __name__ == '__main__':
    d = 30

    arr = np.random.randint(10, size=(d,d))

    dct_C = dct_factor(arr)
    decoded = inv_dct(dct_C)
    # decoded = np.array(decoded, dtype=np.uint8)
    print((arr - decoded).sum()/d**2)
    # pass

