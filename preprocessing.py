import numpy as np
import cv2

def preprocess_image(image, source='pil'):
    if source == 'pil':
        image = np.array(image)

    # Konversi ke BGR jika image masih RGBA atau grayscale
    if len(image.shape) == 2:  # sudah grayscale
        gray = image
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)
