import numpy as np
import cv2


# Histogram of Oriented Gradients
def to_hog(flatten_im, shape_im=(150, 150, 3)):
    """Returns a Histogram of Oriented Gradients vector for the image."""
    im = flatten_im.reshape(shape_im)
    win_size=(150, 150)
    block_size = (10, 10) 
    block_stride = (10, 10)
    cell_size = (5, 5)
    n_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    hist = hog.compute(im)
    return hist


# Color Histogram
def to_color_hist(flatten_im, shape_im =(150, 150, 3)):
    """Returns a Color Histogram vector for the image."""
    im = flatten_im.reshape(shape_im)
    hists = []
    im_channels = cv2.split(im)
    for i, channel in enumerate(['b', 'g', 'r']):
        hists.append(cv2.calcHist(im_channels, [i], None, [64], [0, 256]))
    hist = np.concatenate(hists, axis=None)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist
