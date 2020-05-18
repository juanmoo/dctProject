import numpy as np
from PIL import Image
from math import prod

'''
Loads image in <path> and transforms it into a single-channel image with grayscale 
values. Luminance is computed according to CIE 1931.
'''
def load_grayscale(path, square=False):
    img = Image.open(path)
    img = np.array(img)

    red, green, blue = img.T
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    if square:
        dim = min(gray.shape)
        gray = gray[:dim, :dim]

    gs_img = np.array([gray] * 3, dtype=np.uint8).T
    gs_img = Image.fromarray(gs_img)

    return gs_img

'''
Loads image in <path> in RGB format and outputs array representing image
in YUV coordinates.
'''
def load_image(path, square=False, yuv=True):
    img = Image.open(path)
    img = np.array(img)

    if square:
        dim = min(img.shape[:-1])
        img = img[:dim, :dim, :]

    if yuv:
        img = rgb_to_yuv(img)

    return img

'''
Transform image representing image in YUV coordinates to
RGB coordinates.
'''
def yuv_to_rgb(img):
    assert(len(img.shape) == 3 and img.shape[2] == 3)

    # Inverse of rgb_to_yuv
    yuv_to_rgb = np.array([[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
                         [-7.15253845e-06, -3.44133129e-01,  1.77200251e+00],
                         [ 1.40199759e+00, -7.14138049e-01,  1.54054674e-05]])
    img = np.array(img, dtype=np.float64)
    
    img[:, :, 1:] -= 128.0
    img = img @ yuv_to_rgb

    return img

'''
Transform image representing image in RGB coordinates to
YUV coordinates.
'''
def rgb_to_yuv(img):
    assert(len(img.shape) == 3 and img.shape[2] == 3)

    rgb_to_yuv = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    
    img = img @ rgb_to_yuv
    img[:, :, 1:] += 128.0

    return img

'''
Creates image object from grayscale array.
'''
def image_from_grayscale_array(arr):
    gs_img = np.array([arr] * 3, dtype=np.uint8).T
    gs_img = Image.fromarray(gs_img)

    return gs_img

'''
Create image from 3-channel array.
'''
def img_from_array(arr, is_rgb=False):
    if not is_rgb:
        arr = yuv_to_rgb(arr)

    img = np.array(arr, dtype=np.uint8)
    img = Image.fromarray(img)

    return img

'''
Calculate the size of an image considering only one color channel in bytes.
'''
def calculate_image_size(img):
    return prod(img.size) * 3

'''
Crops image to square and resizes to square image with dimension 'size'.
'''
def square_resize(img, size):
    dim = min(img.size)
    new_image = img.crop((0, 0, dim, dim))
    return new_image.resize((size, size))


'''
Diagonally Traverse 2d array
'''
def diagonalOrder(matrix) : 
    m, n = matrix.shape
    output = []
    for line in range(1, m+n) : 
        start_col = max(0, line - m) 
        count = min(line, (n - start_col), m) 
        for j in range(0, count) : 
            output.append(matrix[min(m, line) - j - 1][start_col + j])
    
    return output

def fillDiagonal(flat, dim):
    out = np.zeros((dim, dim))

    if len(flat) == 0:
        return out
        
    m, n = dim, dim
    c = 0
    for line in range(1, m+n) : 
        start_col = max(0, line - m) 
        count = min(line, (n - start_col), m) 
        for j in range(0, count) : 
            try:
                out[min(m, line) - j - 1][start_col + j] = flat[c]
                c += 1
            except:
                print(len(flat))
                print(out.shape)
    return out

if __name__ == '__main__':
    # image_path = '../tmp/backyard.jpg'
    # img = load_yuv(image_path, square=False)
    # img = img_from_array(img, is_rgb=False)

    # img.save('../tmp/gs.jpg')
    # print(img)

    arr = np.random.randint(10, size=(3, 3))
    print(arr)

    zigzag = diagonalOrder(arr)
    print(zigzag)

    rebuild = fillDiagonal(zigzag, 3)
    print(rebuild)
