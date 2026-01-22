import cv2
import numpy as np

def process_image(img):
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    denoised = cv2.GaussianBlur(gray, (5,5), 0)

    # 3. Edges
    edges = cv2.Canny(denoised, 50, 150)

    # 4. Band enhancement
    bg = cv2.morphologyEx(
        denoised,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    )

    band_only = cv2.subtract(denoised, bg)
    band_only = cv2.normalize(band_only, None, 0, 255, cv2.NORM_MINMAX)

    bands = cv2.adaptiveThreshold(
        band_only, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 3
    )

    return gray, denoised, edges, bands
