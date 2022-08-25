import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable


def collect_ims_by_classes(path):
    """Collects images from classes' directories. 
    A directory from path variable should have the structure: 
    - path
        - class1_name
            - im1.jpg
            - im2.jpg
            - ...
        - class2-name
            - im1.jpg
            - ...
        - ...
    
    Args:
        path (str): A path to directory with classes directories
          with images in them.

    Returns:
        A dict: 
          key: class name
          value: list of images that belong to the class
    """
    class_to_ims = {}
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        ims = [cv2.imread(os.path.join(class_path, im_name)) for im_name in os.listdir(class_path)]
        class_to_ims[class_name] = ims
    return class_to_ims


def resize_im(im, dsize):
    """
    Args:
        im (np.array): An image to resize
        dsize (list): A list [desired width, desired length]

    Returns:
        Resized image
    """
    if im.shape != (dsize[0], dsize[1], 3):
        im = cv2.resize(im, dsize=dsize)
    return im


def create_ims_and_labels_arrays(class_to_ims, class_id, dsize=(150, 150)):
    """ Creates an array with images data and labels.
    Shuffles the samples and resizes images to the common size.
    
    Args:
        class_to_ims (dict): A dict: class name -> a list with class images
        class_id (dict): A dict: class name -> class id
        dsize (tuple): Desired size of an image. Default: (150, 150)

    Returns:
        A pair of np.arrays: shuffled samples and shuffled labels
    """
    samples = []
    labels = []
    for class_name, class_ims in class_to_ims.items():
        class_ims = [resize_im(im, dsize) for im in class_ims]
        samples.append(np.stack(class_ims, 0))
        labels.append(np.array([class_id[class_name]] * len(class_ims)))
    samples = np.concatenate(samples)
    labels = np.concatenate(labels)
    n_samples = len(samples)
    ids = np.array(range(n_samples))
    np.random.shuffle(ids)
    shuffled_samples, shuffled_labels = samples[ids], labels[ids]
    return shuffled_samples, shuffled_labels


# array -> torch vars
def create_variables(X, y):
    "Converts X and y arrays into the torch Variables."
    X_v = Variable(torch.tensor(X, dtype=torch.float32))
    y_v = Variable(torch.tensor(y, dtype=torch.long))
    return X_v, y_v
    

def create_train_val_variables(X, y, n_train):
    "Splits X and y into training and validation sets. Converts them into the torch Variables."
    X_train, y_train = create_variables(X[:n_train, :], y[:n_train])
    X_val, y_val = create_variables(X[n_train:, :], y[n_train:])
    return X_train, X_val, y_train, y_val
