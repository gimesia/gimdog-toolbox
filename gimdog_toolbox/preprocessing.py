import cv2
import numpy as np
import torch

import torch
import numpy as np


def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor into a NumPy image array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (H, W), (C, H, W), or (N, C, H, W).

    Returns
    -------
    np.ndarray
        Converted NumPy array with shape (H, W), (H, W, C), or (H, W, C) after removing batch dimension.

    Raises
    ------
    ValueError
        If the input tensor has an invalid shape.
    """
    match tensor.ndim:
        case 2:
            return tensor.cpu().numpy()
        case 3:
            return tensor.permute(1, 2, 0).cpu().numpy()
        case 4:
            return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        case _:
            raise ValueError("Invalid tensor shape")


def image2tensor(image: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy image array into a PyTorch tensor.

    Parameters
    ----------
    image : np.ndarray
        Input image array with shape (H, W), (H, W, C), or (N, H, W, C).

    Returns
    -------
    torch.Tensor
        Converted PyTorch tensor with shape (1, H, W), (C, H, W), or (N, C, H, W).

    Raises
    ------
    ValueError
        If the input image has an invalid shape.
    """
    match image.ndim:
        case 2:
            return torch.from_numpy(image).unsqueeze(0)
        case 3:
            return torch.from_numpy(image).permute(2, 0, 1)
        case 4:
            return torch.from_numpy(image).permute(0, 3, 1, 2)
        case _:
            raise ValueError("Invalid image shape")


def normalize2uint8(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Normalize an image or tensor to 8-bit unsigned integer type.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image or tensor.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized image or tensor.
    """
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = tensor2image(image)
    normalized_image = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    if is_tensor:
        normalized_image = image2tensor(normalized_image)
    return normalized_image


def normalize2int8(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Normalize an image or tensor to 8-bit signed integer type.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image or tensor.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized image or tensor.
    """
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = tensor2image(image)
    normalized_image = cv2.normalize(
        image,
        None,
        alpha=-128,
        beta=127,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8S,
    )
    if is_tensor:
        normalized_image = image2tensor(normalized_image)
    return normalized_image


def normalize2float(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Normalize an image or tensor to 32-bit floating point type.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image or tensor.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized image or tensor.
    """
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = tensor2image(image)
    normalized_image = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    if is_tensor:
        normalized_image = image2tensor(normalized_image)
    return normalized_image


def normalize2uint16(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Normalize an image or tensor to 16-bit unsigned integer type.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image or tensor.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized image or tensor.
    """
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = tensor2image(image)
    normalized_image = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=65535,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_16U,
    )
    if is_tensor:
        normalized_image = image2tensor(normalized_image)
    return normalized_image


def normalize2int16(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Normalize an image or tensor to 16-bit signed integer type.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image or tensor.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized image or tensor.
    """
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = tensor2image(image)
    normalized_image = cv2.normalize(
        image,
        None,
        alpha=-32768,
        beta=32767,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_16S,
    )
    if is_tensor:
        normalized_image = image2tensor(normalized_image)
    return normalized_image
