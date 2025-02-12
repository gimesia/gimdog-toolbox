import cv2
import numpy as np
import torch

def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).cpu().numpy()


def image2tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)


def normalize2uint8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def normalize2int8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        image,
        None,
        alpha=-128,
        beta=127,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8S,
    )


def normalize2float(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )


def normalize2uint16(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=65535,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_16U,
    )


def normalize2int16(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        image,
        None,
        alpha=-32768,
        beta=32767,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_16S,
    )
