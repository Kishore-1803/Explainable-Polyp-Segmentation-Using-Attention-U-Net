import cv2
import numpy as np
from PIL import Image

class AdvancedImageAnalyzer:
    """
    Advanced preprocessing analysis including Laplacian blur detection,
    color distribution metrics, and quality filtering.
    """
    
    @staticmethod
    def is_blurred(image: Image.Image, threshold: float = 100.0) -> bool:
        """
        Detects if an image is blurry using the variance of the Laplacian.
        Args:
            image (PIL.Image): Input image.
            threshold (float): Variance threshold below which an image is considered blurry.
        Returns:
            bool: True if image is blurred, False otherwise.
        """
        # Convert PIL Image to OpenCV Grayscale
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Compute Variance of Laplacian
        laplacian_var = cv2.Laplacian(cv_image, cv2.CV_64F).var()
        
        return laplacian_var < threshold

    @staticmethod
    def compute_color_statistics(image: Image.Image) -> dict:
        """
        Computes basic color statistics for quality filtering and color normalization heuristics.
        """
        img_array = np.array(image)
        
        # Mean and Std per channel
        means = np.mean(img_array, axis=(0, 1))
        stds = np.std(img_array, axis=(0, 1))
        
        # Check for severe under-exposure (very dark images)
        is_underexposed = np.all(means < 40)
        
        # Check for severe over-exposure (glare dominance)
        is_overexposed = np.all(means > 215)
        
        return {
            "mean_R": means[0], "mean_G": means[1], "mean_B": means[2],
            "std_R": stds[0], "std_G": stds[1], "std_B": stds[2],
            "is_underexposed": is_underexposed,
            "is_overexposed": is_overexposed
        }

    @staticmethod
    def apply_color_normalization(image: Image.Image) -> Image.Image:
        """
        Placeholder for advanced color normalization (e.g. Reinhard or Macenko).
        Currently applies a simple histogram equalization to the Value channel in HSV.
        """
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        
        # Equalize the Value (brightness) channel
        cv_image[:, :, 2] = cv2.equalizeHist(cv_image[:, :, 2])
        
        # Convert back to RGB and PIL
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb_image)
