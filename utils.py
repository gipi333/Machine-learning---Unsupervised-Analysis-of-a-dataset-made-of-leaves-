import os
from distinctipy import distinctipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def load_data(path="./dataset"):
    """
    Can be used to load the dataset
    """

    X = []
    images = [x for x in os.listdir(path) if ".jpg" in x.lower()]
    for image in images:
        temp = plt.imread(os.path.join(path, image))
        X.append(temp)

    X = np.asarray(X).reshape(-1, 128, 72, 3) / 255.0

    return X


def imscatter(x, y, images, ax=None, zoom=0.1):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    for i, image in enumerate(images):

        im = OffsetImage(image, zoom=zoom)
        _x, _y = np.atleast_1d(x[i], y[i])
        artists = []
        for x0, y0 in zip(_x, _y):
            ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([_x, _y]))
        ax.autoscale()

    return artists


def get_colors(labels):

    cmap = []
    if np.min(labels) == -1:
        colors = distinctipy.get_colors(int(np.max(labels) + 2))
        colors[0] = "black"

        for label in labels:
            cmap.append(colors[label + 1])
    else:
        colors = distinctipy.get_colors(int(np.max(labels) + 1))

        for label in labels:
            cmap.append(colors[label])

    return cmap
