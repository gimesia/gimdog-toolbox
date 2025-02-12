import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import torch

def show_image(image, title=None):
    """
    Display a single image using matplotlib.

    Parameters
    ----------
    image : ndarray
        Image to display.
    title : str, optional
        Title for the image. Default is None.
    """
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_images(images, num_cols=4, titles=None):
    """
    Display a list of images in a grid format using matplotlib.

    Parameters
    ----------
    images : list of ndarray
        List of images to display.
    num_cols : int, optional
        Number of columns in the grid. Default is 4.
    titles : list of str, optional
        List of titles for each image. Default is None.
    """
    num_images = max(len(images), num_cols)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_images):
        image = images[i]
        axes[i].imshow(image)

        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])

        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def show_images_with_reference(images, single_image=None, num_cols=4, titles=None):
    """
    Display a grid of images with an optional reference image in each row.

    Parameters
    ----------
    images : list of ndarray
        List of images to display.
    single_image : ndarray, optional
        A single reference image to display in the first column of each row. Defaults to None.
    num_cols : int, optional
        Number of columns of images to display (excluding the reference image column). Defaults to 4.
    titles : list of str, optional
        List of titles for each image. Defaults to None.
    """

    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols + 1, figsize=(15 + 5, num_rows * 3))
    axes = axes.flatten()

    for row in range(num_rows):
        if single_image is not None:
            if isinstance(single_image, torch.Tensor):
                single_image = single_image.permute(1, 2, 0).numpy()
            axes[row * (num_cols + 1)].imshow(single_image)
            axes[row * (num_cols + 1)].set_title("Ref Image")
            axes[row * (num_cols + 1)].axis("off")
        else:
            axes[row * (num_cols + 1)].axis("off")

        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < num_images:
                image = images[idx]
                axes[row * (num_cols + 1) + col + 1].imshow(image)

                if titles is not None and idx < len(titles):
                    axes[row * (num_cols + 1) + col + 1].set_title(titles[idx])

                axes[row * (num_cols + 1) + col + 1].axis("off")
            else:
                axes[row * (num_cols + 1) + col + 1].axis("off")

    plt.tight_layout()
    plt.show()
