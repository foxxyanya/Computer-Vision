import os
import cv2
import matplotlib.pyplot as plt


def show_test_results(path='output_data'):
    fig, axs = plt.subplots(4, 2, figsize=(10, 22))
    for im_idx, im_name in enumerate(os.listdir(path)):
        im_path = os.path.join(path, im_name)
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        i, j = im_idx % 4, im_idx // 4
        axs[i, j].imshow(im)
        axs[i, j].axis("off")
    fig.tight_layout()
    plt.show();
