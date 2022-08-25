import matplotlib.pyplot as plt
import numpy as np


def visualize_linear_regression(model, class_l, class_n=6):
    "For each class, visualizes the corresponding vector from linear layer matrix."
    W = model.state_dict()['linear.weight'].cpu().detach().numpy()
    
    for class_i in range(len(class_l)):
        plt.subplot(1, class_n, class_i+1)
        # weights vector that correspond to the class
        W_class = W[class_i, :]
        # vector back to image shape 
        template = W_class.reshape((150, 150, -1))
        # scaling to [0, 1]
        template = (template - np.min(template)) / (np.max(template) - np.min(template))
        im = template[:, :, [0, 1, 2]]
        class_name = class_l[class_i].capitalize()
        plt.imshow(im)
        plt.title(class_name)
        plt.axis(False);
