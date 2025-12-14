#==============================================================================
#
#
#                           detect_squares.py
#
#
#==============================================================================

# Import des bibliothèques
import cv2
import numpy as np

ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_100
# Création de masques pour cacher les arucos
def Construction_masques_aruco(image, margin_px=40):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

# Passage en niveau de  gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Détection des arucos
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
# Détection des marqueurs : corners = coins détectés, ids = identifiants détectés
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return mask

    for c in corners:
        pts = c.reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)

    if margin_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*margin_px+1, 2*margin_px+1))
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


def carré(approx, contour_area, angle_tol_cos=0.30, aspect_tol=0.15, fill_min=0.60):
    pts = approx.reshape(4, 2).astype(np.float32)
    if not cv2.isContourConvex(pts.reshape(-1,1,2)):
        return False

    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    if w == 0 or h == 0:
        return False
    ratio = w / float(h)
    if not (1.0 - aspect_tol <= ratio <= 1.0 + aspect_tol):
        return False

    rect_area = w * h
    fill_ratio = contour_area / float(rect_area + 1e-9)
    if fill_ratio < fill_min:
        return False

 # Calcule le cosinus de l'angle au point p1
    def angle_cos(p0, p1, p2):
        v1 = p0 - p1
        v2 = p2 - p1
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return abs(num / den)

    idx = np.argsort(pts[:,1], kind="mergesort")
    top = pts[idx[:2]][np.argsort(pts[idx[:2],0])]
    bottom = pts[idx[2:]][np.argsort(pts[idx[2:],0])]
    TL, TR = top[0], top[1]
    BL, BR = bottom[0], bottom[1]

    cosines = [
        angle_cos(TR, TL, BL),
        angle_cos(TL, TR, BR),
        angle_cos(TR, BR, BL),
        angle_cos(BR, BL, TL),
    ]
    return all(c <= angle_tol_cos for c in cosines)


def detect_squares(
    image,
    target_side_mm=20.0,
    px_per_mm=None,
    exclusion_aruco=True,
    aruco_margin=40,
    show=False,
    exclusion_mask=None  # <-- NOUVEAU : masque global
):
    if image is None:
        return {"carré": []}

    h, w = image.shape[:2]
    # priorité au masque global s'il est fourni
    if exclusion_mask is not None:
        aruco_mask = exclusion_mask
    else:
        aruco_mask = Construction_masques_aruco(image, aruco_margin) if exclusion_aruco else np.zeros((h,w), np.uint8)

    # Fenêtre d'aire
    if px_per_mm and px_per_mm > 0:
        side_px = float(px_per_mm) * float(target_side_mm)
        min_area = int((0.55 * side_px) ** 2)
        max_area = int((1.85 * side_px) ** 2)
    else:
        min_area, max_area = 200, 3500

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_) > 127:
        bin_ = cv2.bitwise_not(bin_)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)

    # Exclure ArUco
    bin_[aruco_mask > 0] = 0

    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    carré = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or (max_area and area > max_area):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if aruco_mask[y:y+bh, x:x+bw].any():
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue

        if not carré(approx, area, angle_tol_cos=0.30, aspect_tol=0.15, fill_min=0.60):
            continue

        pts = approx.reshape(4, 2)
        cx, cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
        carré.append({"x": cx, "y": cy, "surface": int(area), "approx": approx})

    return {"carré": carré, "debug": {"bin": bin_, "mask": aruco_mask}}
