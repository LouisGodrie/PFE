#==============================================================================
#
#
#                           masque_aruco.py
#
#
#==============================================================================

# Import bibliothèques
import cv2
import numpy as np

# Dictionnaire Aruco utilisé (selon taille utilisée)
ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_100

# Construction d'un masque unit8 pour masquer les aruco post vision
def construction_masque_aruco(image, margin_px: int = 20) -> np.ndarray:

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
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

# Deploiement du masque et application
def deploiement_masque(image, exclusion_mask: np.ndarray):

    if exclusion_mask is None:
        return image
    if image is None:
        return image

    out = image.copy()
    if out.ndim == 2:
        out[exclusion_mask > 0] = 0
    else:
        out[exclusion_mask > 0] = (0, 0, 0)
    return out
