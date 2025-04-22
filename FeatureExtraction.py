import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from mahotas.features import zernike_moments
import matplotlib.pyplot as plt

def get_hu_moments(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return huMoments.flatten()

def get_zernike_moments(image, radius=21, degree=8):
    return zernike_moments(image, radius, degree=degree)

def get_geometric_moments(image):
    moments = cv2.moments(image)
    return np.array([moments[key] for key in sorted(moments.keys())])

def get_affine_moments(image):
    moments = cv2.moments(image)
    affine_moments = [
        moments['nu20'],
        moments['nu11'],
        moments['nu02'],
        moments['nu30'],
        moments['nu21'],
        moments['nu12'],
        moments['nu03'],
    ]
    return np.array(affine_moments)

def get_legendre_moments(image, order=3):
    # Normalize image
    image = image.astype(np.float32) / 255.0
    x = np.linspace(-1, 1, image.shape[1])
    y = np.linspace(-1, 1, image.shape[0])
    X, Y = np.meshgrid(x, y)

    def legendre(n, x):
        if n == 0: return np.ones_like(x)
        elif n == 1: return x
        else: return ((2*n - 1)*x*legendre(n-1, x) - (n-1)*legendre(n-2, x))/n

    moments = []
    for p in range(order+1):
        for q in range(order+1):
            Lp = legendre(p, X)
            Lq = legendre(q, Y)
            moment = np.sum(image * Lp * Lq)
            moments.append(moment)
    return np.array(moments)

def get_orthogonal_moments(image):
    # Use Zernike again or define other orthogonal types
    return get_zernike_moments(image)

def get_complex_moments(image):
    moments = cv2.moments(image)
    c_moments = []
    for p in range(4):
        for q in range(4 - p):
            cm = np.sum((np.indices(image.shape)[0]**p) *
                        (np.indices(image.shape)[1]**q) * image)
            c_moments.append(cm)
    return np.array(c_moments)

def extract_all_moments(image_path):
    image = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    features = {
        "Hu": get_hu_moments(thresh),
        "Zernike": get_zernike_moments(thresh),
        "Geometric": get_geometric_moments(thresh),
        "Affine": get_affine_moments(thresh),
        "Legendre": get_legendre_moments(thresh),
        "Orthogonal": get_orthogonal_moments(thresh),
        "Complex": get_complex_moments(thresh),
    }
    return features

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python moments_extraction.py path_to_image")
    else:
        image_path = sys.argv[1]
        feats = extract_all_moments(image_path)
        for key, value in feats.items():
            print(f"{key} Moments:\n{value}\n")
