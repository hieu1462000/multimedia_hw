import math
import numpy as np

def cos(k, n, N):
    return math.cos(math.pi * k * (2 * n + 1) / (2 * N))


def dct(F):
    N = F.shape[0]
    output = np.zeros_like(F)

    for k in range(N):
        s = 1 if k != 0 else math.sqrt(0.5)
        alpha = math.sqrt(2 / N) * s
        sum = 0
        for n in range(N):
            sum += F[n] * cos(k, n, N)

        output[k] = alpha * sum

    return output

def idct(F):
    N = F.shape[0]
    output = np.zeros_like(F)

    for n in range(N):
        sum = 0
        for k in range(N):
            s = 1 if k != 0 else math.sqrt(0.5)
            alpha = math.sqrt(2 / N) * s
            sum += alpha * F[k] * cos(k,n,N)

        output[n] = sum

    return output


def dct2(matrix):
    H, W = matrix.shape[0], matrix.shape[1]
    # Apply 1D DCT (Horizontally) to rows
    # Apply 1D DCT (Vertically) to resultant Horizontally DCT above.
    result = np.transpose(dct(np.transpose(dct(matrix))))
    
    # Take only the first quadrant
    output = np.zeros_like(result)
    output[0: H // 2, 0: W // 2] = result[0: H // 2, 0: W // 2]
    return output


def idct2(matrix):
    result = np.transpose(idct(np.transpose(idct(matrix))))
    return result



##############################
bl_size = 8


def addPadding(img):
    H, W = img.shape[0], img.shape[1]
    h_blocks = H / bl_size if H % bl_size == 0 else H // bl_size + 1
    w_blocks = W / bl_size if W % bl_size == 0 else W // bl_size + 1
    padded_h = int(h_blocks * bl_size)
    padded_w = int(w_blocks * bl_size)

    result = np.zeros((padded_h, padded_w), dtype=img.dtype)
    result[0:H, 0:W] = img
    return result, padded_h, padded_w


def img2dct(img):
    H, W = img.shape[0], img.shape[1]
    padded_img, padded_h, padded_w = addPadding(img)
    dct_img = np.zeros_like(padded_img)

    for row in np.arange(padded_h - bl_size + 1, step=bl_size):
        for col in np.arange(padded_w - bl_size + 1, step=bl_size):
            dct_matrix = dct2(padded_img[row: row+bl_size, col: col+bl_size])
            dct_img[row: row+bl_size, col: col+bl_size] = dct_matrix

    return dct_img[0:H, 0:W]


def dct2img(dct_img):
    H, W = dct_img.shape[0], dct_img.shape[1]
    padded_img, padded_h, padded_w = addPadding(dct_img)
    img = np.zeros_like(padded_img)

    for row in np.arange(padded_h - bl_size + 1, step=bl_size):
        for col in np.arange(padded_w - bl_size + 1, step=bl_size):
            dct_matrix = idct2(padded_img[row: row+bl_size, col: col+bl_size])
            img[row: row+bl_size, col: col+bl_size] = dct_matrix

    return img[0:H, 0:W]