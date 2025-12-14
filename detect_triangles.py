#==============================================================================
#
#
#                           detect_triangles.py
#
#
#==============================================================================

# Import des bibliothèques
import cv2
import numpy as np

def detect_triangles(image, exclusion_mask=None):
    if image is None:
        return {"triangles": []}

    # applique masque si fourni
    if exclusion_mask is not None:
        img = image.copy()
        img[exclusion_mask > 0] = (0, 0, 0)
    else:
        img = image

# Niveaux de gris + flou pour réduire le bruit
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Seuillage auto
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_) > 127:
        bin_ = cv2.bitwise_not(bin_)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)

    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tris = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

# Appro on garde uniquement les contours à 3 sommets 
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
        if len(approx) != 3:
            continue

        pts = approx.reshape(3, 2).astype(np.float32)
        if not cv2.isContourConvex(pts.reshape(-1,1,2)):
            continue

        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        fill_ratio = area / float(max(1, w*h))
        if fill_ratio < 0.35:
            continue

# Centre du triangle (moyenne des sommets)
        cx, cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
        tris.append({"x": cx, "y": cy, "approx": approx})

    return {"triangles": tris, "debug": {"bin": bin_}}
