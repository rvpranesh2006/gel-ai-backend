import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128

# Load model WITHOUT compiling (fixes Keras version issue)
model = tf.keras.models.load_model(
    "model/autoencoder.h5",
    compile=False
)

def detect_anomaly(band_img):
    img = cv2.resize(band_img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    recon = model.predict(img, verbose=0)

    error = np.mean((img - recon) ** 2)
    heatmap = np.abs(img - recon)[0]

    return error, recon[0], heatmap
