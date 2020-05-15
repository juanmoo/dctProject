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
Creates image object from grayscale array.
'''
def image_from_grayscale_array(arr):
    gs_img = np.array([arr] * 3, dtype=np.uint8).T
    gs_img = Image.fromarray(gs_img)

    return gs_img

'''
Calculate the size of an image considering only one color channel in bytes.
'''
def calculate_image_size(img):
    return prod(img.size) * 3

'''
Crops image to square and resizes to square image with size 512 by 512 px
'''
def square_resize(img, size):
    dim = min(img.size)
    new_image = img.crop((0, 0, dim, dim))
    return new_image.resize((size, size))


if __name__ == '__main__':
    image_path = '../tmp/backyard.jpg'
    img = load_grayscale(image_path, square=False)
    img.save('../tmp/gs.jpg')
    print(img)


    resized = resize512(img)
    resized.save('../tmp/resized.jpg')
