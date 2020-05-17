import heapq
import os
from functools import total_ordering


class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if(other == None):
            return False
        if(not isinstance(other, HeapNode)):
            return False
        return self.freq == other.freq


class HuffmanCoding:
    def __init__(self, symbols):
        self.symbols = symbols
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # functions for compression:
    def make_frequency_dict(self, symbols):
        frequency = {}
        for s in symbols:
            if not s in frequency:
                frequency[s] = 0
            frequency[s] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)


    def make_codes_helper(self, root, current_code):
        if(root == None):
            return

        if(root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")


    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)


    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text


    def compress(self):

        frequency = self.make_frequency_dict(self.symbols)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded_text = self.get_encoded_text(self.symbols)

        return encoded_text


    """ functions for decompression: """

    def decode_text(self, encoded_text):
        current_code = ""
        decoded = []

        for bit in encoded_text:
            current_code += bit
            if(current_code in self.reverse_mapping):
                s = self.reverse_mapping[current_code]
                decoded.append(s)
                current_code = ""

        return decoded


    def decompress(self, encoded_text):
        decompressed_text = self.decode_text(encoded_text)
        return decompressed_text


if __name__ == '__main__':
    path = [32.4, 1, 1, 2, 3, 4, 4, 4, 4, 4]

    h = HuffmanCoding(path)

    output = h.compress()
    print("Compressed:", output)

    decom = h.decompress(output)
    print("Decompressed:", decom)