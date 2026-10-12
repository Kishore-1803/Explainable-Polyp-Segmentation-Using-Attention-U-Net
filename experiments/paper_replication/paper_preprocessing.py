"""
Preprocessing variants for the paper-replication study.
=======================================================

Two deterministic preprocessing "arms" are defined here. Both are applied
identically to training, validation and test data (Kvasir-SEG and CVC-ClinicDB),
exactly as the base paper states its pipeline is applied to "both training and
evaluation datasets".

ARM "paper"  -- follows Gupta & Benila (IEEE Access 2025), Sec. IV-A-3:
    1. Resolution standardisation ....... resize to 256x256
    2. Min-max intensity normalisation .. rescale each image to [0, 255]
    3. Histogram equalisation ........... channel-wise on RGB
    4. CLAHE ............................ local contrast adjustment
    5. Gaussian filter .................. sigma = 1.0
    6. Median filter .................... 3x3 kernel

ARM "ours"   -- our modification:
    1. Resolution standardisation ....... resize to 256x256
    2. CLAHE on the CIE-LAB L* channel .. clip 2.0, 8x8 tiles (chroma untouched)
    (+ the blur / exposure quality gate, which flags frames but does not alter
       pixels; flags are recorded in the cache manifest.)

DOCUMENTED DEVIATIONS (the paper does not specify these):
  * CLAHE clip limit and tile grid are not given in the paper. We use the same
    values as our own pipeline (clip 2.0, 8x8) so the two arms differ only in
    WHERE contrast equalisation is applied, not in how strong it is.
  * The paper applies CLAHE after channel-wise histogram equalisation but does
    not say in which colour space. We apply it on the LAB L* channel, the
    standard choice for colour images.
  * ImageNet channel normalisation is retained as the final step in BOTH arms
    (it is part of our model's input contract and the paper does not exclude
    it). Min-max to [0,1] is implicit in uint8 -> float conversion.
"""

import cv2
import numpy as np

IMAGE_SIZE = (256, 256)          # (width, height) for cv2
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
BLUR_VAR_THRESHOLD = 80.0        # Laplacian variance below this => blurred
EXPOSURE_LOW = 0.15              # mean brightness outside [15%, 85%] => flagged
EXPOSURE_HIGH = 0.85


def _resize_image(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)


def _min_max_normalise(img: np.ndarray) -> np.ndarray:
    """Rescale the whole image to the full [0, 255] range (paper eq. 9)."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - lo) / (hi - lo)).astype(np.uint8)


def _hist_equalise_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Standard histogram equalisation applied channel-wise."""
    out = np.empty_like(img_rgb)
    for c in range(3):
        out[:, :, c] = cv2.equalizeHist(img_rgb[:, :, c])
    return out


def _clahe_on_luminance(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the CIE-LAB L* channel only; a* and b* are left untouched."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _denoise(img_rgb: np.ndarray) -> np.ndarray:
    """Gaussian (sigma=1.0) then 3x3 median, as specified in the paper."""
    img = cv2.GaussianBlur(img_rgb, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.medianBlur(img, 3)


def quality_flags(img_rgb: np.ndarray) -> dict:
    """Blur / exposure gate. Reports only -- it does not modify the image."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean() / 255.0)
    return {
        "laplacian_var": lap_var,
        "mean_brightness": brightness,
        "blurred": bool(lap_var < BLUR_VAR_THRESHOLD),
        "exposure_outlier": bool(brightness < EXPOSURE_LOW or brightness > EXPOSURE_HIGH),
    }


def preprocess_paper(img_rgb: np.ndarray) -> np.ndarray:
    """ARM 'paper': the base paper's full preprocessing chain."""
    img = _resize_image(img_rgb)
    img = _min_max_normalise(img)
    img = _hist_equalise_rgb(img)
    img = _clahe_on_luminance(img)
    img = _denoise(img)
    return img


def preprocess_ours(img_rgb: np.ndarray) -> np.ndarray:
    """ARM 'ours': luminance-only CLAHE, no global equalisation, no filtering."""
    img = _resize_image(img_rgb)
    img = _clahe_on_luminance(img)
    return img


PREPROCESSORS = {
    "paper": preprocess_paper,
    "ours": preprocess_ours,
}
