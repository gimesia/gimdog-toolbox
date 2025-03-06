import copy

import cv2
import numpy as np
import skimage
import torch
from matplotlib import pyplot as plt

from .preprocessing import normalize2uint8, tensor2image


def show_image(image, title=None, cmap=None):
    """
    Display a single image using matplotlib.

    Parameters
    ----------
    image : ndarray
        Image to display.
    title : str, optional
        Title for the image. Default is None.
    cmap : str, optional
        Colormap to use for displaying the image. Default is None, which uses the default colormap.
    """
    # Create a deep copy of the image to avoid modifying the original
    image_copy = copy.deepcopy(image)

    # Convert torch.Tensor to numpy array if necessary
    if isinstance(image_copy, torch.Tensor):
        image_copy = tensor2image(image_copy)

    # Ensure the image is in uint8 format
    if image_copy.dtype != np.uint8:
        image_copy = normalize2uint8(image_copy)

    # Display the image using matplotlib
    plt.figure()
    plt.imshow(image_copy, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_images(images, num_cols=4, titles=None, cmap=None):
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
    cmap : str, optional
        Colormap to use for displaying the images. Default is None.
    """
    # Create a deep copy of the images to avoid modifying the originals
    images_copy = copy.deepcopy(images)
    num_images = max(len(images_copy), num_cols)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_images):
        try:
            image = images_copy[i]

            # Convert torch.Tensor to numpy array if necessary
            if isinstance(image, torch.Tensor):
                image = tensor2image(image)

            # Ensure the image is in uint8 format
            if image.dtype != np.uint8:
                image = normalize2uint8(image)

            # Display the image in the subplot
            axes[i].imshow(image, cmap=cmap)

            # Set the title if provided
            if titles is not None and i < len(titles):
                axes[i].set_title(titles[i])

            axes[i].axis("off")
        except Exception as e:
            print(f"Error displaying image {i}: {e}")

    # Turn off any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def show_images_with_reference(
    images: list[np.ndarray | torch.Tensor],
    single_image: np.ndarray | torch.Tensor | None = None,
    num_cols: int = 4,
    titles: list[str] | None = None,
    cmap: str | None = None,
) -> None:
    """
    Display a grid of images with an optional reference image in each row.

    Parameters
    ----------
    images : list of ndarray or torch.Tensor
        List of images to display.
    single_image : ndarray or torch.Tensor, optional
        A single reference image to display in the first column of each row. Defaults to None.
    num_cols : int, optional
        Number of columns of images to display (excluding the reference image column). Defaults to 4.
    titles : list of str, optional
        List of titles for each image. Defaults to None.
    cmap : str, optional
        Colormap to use for displaying the images. Default is None.
    """
    # Create deep copies of the images to avoid modifying the originals
    images_copy = copy.deepcopy(images)
    single_image_copy = (
        copy.deepcopy(single_image) if single_image is not None else None
    )

    num_images = len(images_copy)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create a grid of subplots with an extra column for the reference image
    fig, axes = plt.subplots(num_rows, num_cols + 1, figsize=(15 + 5, num_rows * 3))
    axes = axes.flatten()

    for row in range(num_rows):
        if single_image_copy is not None:
            # Convert torch.Tensor to numpy array if necessary
            if isinstance(single_image_copy, torch.Tensor):
                single_image_copy = tensor2image(single_image_copy)

            # Ensure the image is in uint8 format
            if single_image_copy.dtype != np.uint8:
                single_image_copy = normalize2uint8(single_image_copy)

            # Display the reference image in the first column of the row
            axes[row * (num_cols + 1)].imshow(single_image_copy, cmap=cmap)
            axes[row * (num_cols + 1)].set_title("Ref Image")
            axes[row * (num_cols + 1)].axis("off")
        else:
            axes[row * (num_cols + 1)].axis("off")

        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < num_images:
                image = images_copy[idx]

                # Convert torch.Tensor to numpy array if necessary
                if isinstance(image, torch.Tensor):
                    image = tensor2image(image)

                # Ensure the image is in uint8 format
                if image.dtype != np.uint8:
                    image = normalize2uint8(image)

                # Display the image in the subplot
                axes[row * (num_cols + 1) + col + 1].imshow(image, cmap=cmap)

                # Set the title if provided
                if titles is not None and idx < len(titles):
                    axes[row * (num_cols + 1) + col + 1].set_title(titles[idx])

                axes[row * (num_cols + 1) + col + 1].axis("off")
            else:
                axes[row * (num_cols + 1) + col + 1].axis("off")

    plt.tight_layout()
    plt.show()


def draw_mask_outlines(image, mask, color=(0, 255, 0), thickness=3):
    """
    Draws the outlines of a mask on an image.
    Parameters:
    image (numpy.ndarray or torch.Tensor): The input image on which to draw the mask outlines.
                                            If a torch.Tensor is provided, it will be converted to a numpy array.
    mask (numpy.ndarray or torch.Tensor): The binary mask indicating the regions to outline.
                                            If a torch.Tensor is provided, it will be converted to a numpy array.
    color (tuple): A tuple representing the RGB color of the mask outlines. Default is green (0, 255, 0).
    Returns:
    numpy.ndarray: The image with the mask outlines drawn on it.
    """
    # Convert torch.Tensor to numpy array if necessary
    if isinstance(image, torch.Tensor):
        outlined_image = tensor2image(image)
    else:
        outlined_image = image.copy()

    # Ensure the image is in uint8 format for matplotlib
    if outlined_image.dtype != np.uint8:
        outlined_image = normalize2uint8(outlined_image)

    # Ensure the outlined image is 3-channel
    if outlined_image.ndim == 2:
        outlined_image = skimage.color.gray2rgb(outlined_image)

    # Convert torch.Tensor to numpy array if necessary
    if isinstance(mask, torch.Tensor):
        mask_copy = tensor2image(mask)
    else:
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask_copy = mask[0].copy()
        else:
            mask_copy = mask.copy()

    # Find contours of the mask
    contours, _ = cv2.findContours(
        normalize2uint8(mask_copy), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(outlined_image, contours, -1, color, thickness)

    return outlined_image


def draw_multiple_masks(image, masks, colors, thickness=3):
    """
    Draw multiple masks on an image.

    Parameters
    ----------
    image : ndarray
        The image on which to draw the masks.
    masks : list of ndarray
        A list of binary masks to draw on the image.
    colors : list of tuple
        A list of colors corresponding to each mask.
    thickness : int, optional
        Thickness of the mask outlines. Default is 3.

    Returns
    -------
    ndarray
        The image with the masks drawn on it.
    """
    # Convert torch.Tensor to numpy array if necessary
    if isinstance(image, torch.Tensor):
        outlined_image = tensor2image(image)
    else:
        outlined_image = image.copy()

    # Ensure the image is in uint8 format for matplotlib
    if outlined_image.dtype != np.uint8:
        outlined_image = normalize2uint8(outlined_image)

    # Draw each mask on the image
    for i, mask in enumerate(masks):
        outlined_image = draw_mask_outlines(outlined_image, mask, colors[i], thickness)

    return outlined_image


def show_image_with_mask_outlines(
    image: np.ndarray | torch.Tensor,
    masks: np.ndarray,
    mask_colors: list[tuple[int]],
    thickness: int = 3,
    cmap: str = "gray",
) -> None:
    """
    Draw mask outlines on an image.

    Parameters
    ----------
    image : ndarray or torch.Tensor
        The image on which to draw the mask outlines.
    masks : ndarray
        A 3D array where each slice along the third axis is a binary mask.
    mask_colors : list of tuples of int
        A list of colors corresponding to each mask.
    thickness : int, optional
    cmap : str, optional
        Colormap to use for displaying the image. Default is None.
    """
    # Convert torch.Tensor to numpy array if necessary
    if isinstance(image, torch.Tensor):
        outlined_image = tensor2image(image)
    else:
        outlined_image = image.copy()

    # Ensure the image is in uint8 format for matplotlib
    if outlined_image.dtype != np.uint8:
        outlined_image = normalize2uint8(outlined_image)

    # Draw the mask outlines on the image
    for i, mask in enumerate(masks):
        outlined_image = draw_mask_outlines(
            outlined_image, mask, mask_colors[i], thickness
        )

    # Display the image with mask outlines
    plt.imshow(outlined_image)
    plt.axis("off")
    plt.show()
