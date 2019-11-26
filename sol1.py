import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage.color import rgb2gray

NORMA_FACTOR = 255  # constant variable for normalizing the image
BINS_NUM_HIST = 257  # The number of bins to create a histogram

# Constants that symbolize display methods
RGB_REPRE = 2
GRAY_REPRE = 1

RGB_SHAPE_LEN = 3  # Number of dimensions in the RGB matrix

# Matrices to convert between RGB and YIQ
RGB_TO_YIQ_MAT = [
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
]
YIQ_TO_RGB_MAT = np.linalg.inv(RGB_TO_YIQ_MAT)


def read_image(filename, representation):
    """
    A function that reads, converts and Normalizes an image by path and representation
    :param filename: Path to Image
    :param representation: Image Style 1 to RGB 2 to Gray
    :return: Converted and normalized image
    """
    im = imread(filename)
    flo_im = im.astype(np.float64)
    norm_im = flo_im / NORMA_FACTOR

    # Converted and normalized representation
    if (representation == GRAY_REPRE) and (len(im.shape) == RGB_SHAPE_LEN):
        norm_im = rgb2gray(im)

    return norm_im


def imdisplay(filename, representation):
    """
    function that display the image
    :param filename: Path to Image
    :param representation: Image Style 1 to RGB 2 to Gray
    """
    im = read_image(filename, representation)
    plt.figure()
    if representation == GRAY_REPRE:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.axis('off')
    plt.show()


def rgb2yiq(imRGB):
    """
    A function that transform an RGB image into the YIQ color space
    :param imRGB: Original image in RGB space
    :return:Image is converted to YIQ space
    """
    # Isolates all color components into separate matrices
    R = imRGB[:, :, 0]
    G = imRGB[:, :, 1]
    B = imRGB[:, :, 2]

    imYIQ = np.copy(imRGB)  # Defines a variable for the new image

    # Calculate the new image according to the conversion matrix
    for i in range(3):
        imYIQ[:, :, i] = R * RGB_TO_YIQ_MAT[i][0] + \
                         G * RGB_TO_YIQ_MAT[i][1] + \
                         B * RGB_TO_YIQ_MAT[i][2]
    return imYIQ


def yiq2rgb(imYIQ):
    """
    A function that transform an YIQ image into the RGB color space
    :param imYIQ: Original image in YIQ space
    :return:Image is converted to RGB space
    """
    # Isolates all color components into separate matrices
    Y = imYIQ[:, :, 0]
    I = imYIQ[:, :, 1]
    Q = imYIQ[:, :, 2]

    imRGB = np.empty(imYIQ.shape)  # Defines a variable for the new image

    # Calculate the new image according to the conversion matrix
    for i in range(3):
        imRGB[:, :, i] = Y * YIQ_TO_RGB_MAT[i, 0] + \
                         I * YIQ_TO_RGB_MAT[i, 1] + \
                         Q * YIQ_TO_RGB_MAT[i, 2]

    return imRGB


def histogram_equalize(im_orig):
    """
    a function that performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: Original image
    :return:List that containing - the post-process image, the original and the edited histograms.
    """
    # Initialize variables for the calculation:
    im_to_equalize = get_im_to_change(im_orig)  # Image for editing
    hist_orig, bounds = np.histogram(im_to_equalize, np.arange(BINS_NUM_HIST))
    hist_sum = np.cumsum(hist_orig)  # Cumulative histogram
    first_non_zero = np.nonzero(hist_sum)[0][0]

    # Making calculations:
    hist_sum_curr = np.round(
        (hist_sum - first_non_zero) / (hist_sum[-1] - first_non_zero) * NORMA_FACTOR)
    new_im = (hist_sum_curr.astype(np.int64))[im_to_equalize.astype(np.int64)]
    new_im = np.clip(new_im / NORMA_FACTOR, 0, 1)

    hist_eq, bounds2 = np.histogram(im_to_equalize, np.arange(BINS_NUM_HIST))
    im_eq = get_final_im(im_orig, new_im)

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    function that performs optimal quantization of a given grayscale or RGB image
    :param im_orig:  input grayscale or RGB image to be quantized
    :param n_quant: number of intensities your output im_quant image should have
    :param n_iter: e maximum number of iterations of the optimization procedure
    :return: List that containing - quantized output image,
                array of total intensities error for each iteration of the quantization procedure
    """
    # Initialize variables for the calculation:
    im_to_quantize = get_im_to_change(im_orig)
    hist_orig, bounds = np.histogram(im_to_quantize, np.arange(BINS_NUM_HIST))
    hist_sum = np.cumsum(hist_orig)

    # Calculate the first z series:
    pix_in_part = np.round(hist_sum[-1] / n_quant)
    z_indexs = np.zeros(n_quant + 1).astype(np.int64)
    for i in range(n_quant):
        z_indexs[i] = np.searchsorted(hist_sum, pix_in_part * i)
    z_indexs[-1] = 255

    q_val = np.zeros(n_quant)
    error_arr = []

    # Perform the iterations:
    for one_iter in range(n_iter):
        q_calculation(hist_orig, n_quant, q_val, z_indexs)
        if z_calculation(n_quant, q_val, z_indexs):
            break
        add_error(error_arr, hist_orig, n_quant, q_val, z_indexs)

    # Calculate the final image
    new_hist = np.zeros(hist_orig.shape)
    for i in range(n_quant):
        np.put(new_hist, range(z_indexs[i], z_indexs[i + 1] + 1), q_val[i])
    new_im = new_hist[im_to_quantize.astype(np.int64)] / NORMA_FACTOR
    im_quant = get_final_im(im_orig, new_im)

    return [im_quant, error_arr]


def get_im_to_change(im_orig):
    """
    A function that creates the image for editing by the image data
    :param im_orig: Original image
    :return: Editable image
    """
    if len(im_orig.shape) == RGB_SHAPE_LEN:
        yiq_im_orig = rgb2yiq(im_orig)
        im_to_chang = yiq_im_orig[:, :, 0] * NORMA_FACTOR
    else:
        im_to_chang = im_orig * NORMA_FACTOR
    return im_to_chang


def get_final_im(im_orig, new_im):
    """
    A function that converts the edited image to the desired format
    :param im_orig: Original image
    :param new_im: The edited image
    :return: ready converted image
    """
    if len(im_orig.shape) == RGB_SHAPE_LEN:
        im_eq_yiq = rgb2yiq(im_orig).copy()
        im_eq_yiq[:, :, 0] = new_im
        im_eq = yiq2rgb(im_eq_yiq)
    else:
        im_eq = new_im
    return im_eq


def add_error(error_arr, hist_orig, n_quant, q_val, z_indexs):
    """
    A function that calculates and adds an error according to a new z and q calculation
    :param error_arr: The current array of errors
    :param hist_orig: The original histogram
    :param n_quant: The number of quantum
    :param q_val: Q values
    :param z_indexs: Z values
    """
    error = 0
    for i in range(n_quant):
        for z in range(z_indexs[i], z_indexs[i + 1]):
            error += ((q_val[i] - z) ** 2) * hist_orig[z]
    error_arr.append(error)


def z_calculation(n_quant, q_val, z_indexs):
    """
    A function that calculates new z values
    :param n_quant: The number of quantum
    :param q_val: Q values
    :param z_indexs: Z values
    :return: True, if nothing else, another False
    """
    finish = True
    for i in range(1, n_quant):
        new_z = np.round((q_val[i - 1] + q_val[i]) / 2)
        if new_z != z_indexs[i]:
            finish = False
        z_indexs[i] = new_z
    return finish


def q_calculation(hist_orig, n_quant, q_val, z_indexs):
    """
    A function that calculates new q values
    :param hist_orig: The original histogram
    :param n_quant: The number of quantum
    :param q_val: Q values
    :param z_indexs: Z values
    """
    for i in range(n_quant):
        sum_fixel_in_q = 0
        sum_val_in_z = 0
        for z in range(z_indexs[i], z_indexs[i + 1] + 1):
            sum_val_in_z += z * hist_orig[z]
            sum_fixel_in_q += hist_orig[z]
        q_val[i] = sum_val_in_z / sum_fixel_in_q


if __name__ == "__main__":
    im = read_image("low_contrast.jpg", RGB_REPRE)
    new = quantize(im, 80, 100)

