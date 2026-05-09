import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def process_image(img):
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (224, 224))
    return img

def extract_features(image):

    # 1. RGB stats
    mean = image.mean(axis=(0,1))
    std = image.std(axis=(0,1))

    # 2. HSV stats
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_mean = hsv.mean(axis=(0,1))
    hsv_std = hsv.std(axis=(0,1))

    # 3. LBP (texture)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # 4. Edges
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # 5. Color Histogram (🔥 مهم جدًا للتمييز)
    hsv_hist = cv2.calcHist([hsv], [0,1], None, [8,8], [0,180,0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # 6. White pixels (تمييز mildew)
    white_pixels = np.sum(
        (image[:,:,0] > 200) &
        (image[:,:,1] > 200) &
        (image[:,:,2] > 200)
    ) / image.size

    return np.concatenate([
        mean, std,
        hsv_mean, hsv_std,
        hist,
        hsv_hist,
        [edge_density],
        [white_pixels]
    ])