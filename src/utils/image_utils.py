"""Image processing utilities."""

import base64
import io
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image
import numpy as np


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from a file path."""
    return Image.open(path)


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """Convert bytes to a PIL Image."""
    return Image.open(io.BytesIO(data))


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to base64 string."""
    return base64.b64encode(image_to_bytes(image, format)).decode("utf-8")


def base64_to_image(data: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    return bytes_to_image(base64.b64decode(data))


def get_image_size(image: Union[Image.Image, bytes, str, Path]) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    if isinstance(image, Image.Image):
        return image.size
    elif isinstance(image, (str, Path)):
        return Image.open(image).size
    elif isinstance(image, bytes):
        return bytes_to_image(image).size
    elif isinstance(image, str):
        try:
            return bytes_to_image(base64.b64decode(image)).size
        except Exception:
            return bytes_to_image(image.encode()).size
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> Image.Image:
    """Resize an image to the target size."""
    if keep_aspect_ratio:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image
    return image.resize(target_size, Image.Resampling.LANCZOS)


def crop_image(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Crop an image given a bounding box (x1, y1, x2, y2)."""
    return image.crop(bbox)


def compute_image_similarity(
    image1: Union[Image.Image, np.ndarray],
    image2: Union[Image.Image, np.ndarray],
) -> float:
    """Compute similarity between two images (1.0 = identical, 0.0 = completely different)."""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    if image1.shape != image2.shape:
        image2_pil = Image.fromarray(image2).resize(
            (image1.shape[1], image1.shape[0]), Image.Resampling.LANCZOS
        )
        image2 = np.array(image2_pil)

    if len(image1.shape) == 3:
        diff = np.abs(image1.astype(float) - image2.astype(float))
        mean_diff = np.mean(diff)
    else:
        diff = np.abs(image1.astype(float) - image2.astype(float))
        mean_diff = np.mean(diff)

    max_diff = 255.0
    similarity = 1.0 - (mean_diff / max_diff)
    return float(similarity)


def compute_pixel_difference(
    image1: Union[Image.Image, np.ndarray],
    image2: Union[Image.Image, np.ndarray],
) -> np.ndarray:
    """Compute per-pixel difference between two images."""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    if image1.shape != image2.shape:
        image2_pil = Image.fromarray(image2).resize(
            (image1.shape[1], image1.shape[0]), Image.Resampling.LANCZOS
        )
        image2 = np.array(image2_pil)

    return np.abs(image1.astype(int) - image2.astype(int)).astype(np.uint8)
