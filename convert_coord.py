#==============================================================================
#
#
#                           convert_coord.py
#
#
#==============================================================================

# Import Bibliothèques
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict

# Longueur et largeur plaque renseignée pour calibration
Plaque_largeur_mm  = 240.0
Plaque_longueur_mm = 260.0

# Dictionnaire Aruco utilisé (selon taille utilisée)
ARUCO_DICT_NAME = cv2.aruco.DICT_5X5_100

_H: Optional[np.ndarray] = None          # homographie pixels->mm
_ROI_IMG: Optional[np.ndarray] = None    # hull convexe pour pointPolygonTest
_PLATE_SIZE_PIX: Optional[Tuple[int,int]] = None  # (h,w) roi

# correction distorsion
_USE_CALIB: bool = False
_K: Optional[np.ndarray] = None
_D: Optional[np.ndarray] = None
_MANUAL_K: float = 0.0

def calibration_camera(K: np.ndarray, D: np.ndarray):
    global _USE_CALIB, _K, _D
    _K = np.asarray(K, dtype=np.float32)
    _D = np.asarray(D, dtype=np.float32).reshape(-1)
    _USE_CALIB = True

def effacement_calibration():
    global _USE_CALIB, _K, _D
    _USE_CALIB, _K, _D = False, None, None

def distorsion_manuel(k: float):
    global _MANUAL_K
    _MANUAL_K = float(k)

def distorsion_kd(frame: np.ndarray) -> np.ndarray:
    if _K is None or _D is None:
        return frame
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(_K, _D, (w, h), alpha=0)
    return cv2.undistort(frame, _K, _D, None, newK)

def distorsion_manuel_k(frame: np.ndarray, k: float) -> np.ndarray:
    if abs(k) < 1e-8:
        return frame
    h, w = frame.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    f = max(h, w)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (xs - cx) / f
    y = (ys - cy) / f
    r2 = x*x + y*y
    scale = 1.0 + k * r2
    map_x = (x * scale * f + cx).astype(np.float32)
    map_y = (y * scale * f + cy).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def repere_distorsion(frame: np.ndarray) -> np.ndarray:
    if _USE_CALIB and _K is not None and _D is not None:
        return distorsion_kd(frame)
    if abs(_MANUAL_K) > 1e-8:
        return distorsion_manuel_k(frame, _MANUAL_K)
    return frame

# ArUco + homographie

def detection_centre_aruco(frame: np.ndarray) -> Dict[int, Tuple[float, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    centers: Dict[int, Tuple[float, float]] = {}
    if ids is None or len(ids) == 0:
        return centers
    ids = ids.flatten().astype(int)
    for i, cid in enumerate(ids):
        pts = corners[i].reshape(-1, 2)  # (4,2)
        c = pts.mean(axis=0)
        centers[int(cid)] = (float(c[0]), float(c[1]))
    return centers

def ordre_aruco(pts: np.ndarray) -> np.ndarray:
    assert pts.shape[0] >= 4
    # On prend les 4 plus "extérieurs" par hull si jamais >4 marqueurs
    if pts.shape[0] > 4:
        hull = cv2.convexHull(pts.reshape(-1,1,2)).reshape(-1,2)
        # garde les 4 sommets principaux
        if hull.shape[0] > 4:
            # approx polygone à 4 sommets
            peri = cv2.arcLength(hull.reshape(-1,1,2), True)
            epsilon = 0.02 * peri
            approx = cv2.approxPolyDP(hull.reshape(-1,1,2), epsilon, True).reshape(-1,2)
            if approx.shape[0] == 4:
                pts = approx
            else:
                pts = hull[:4]
        else:
            pts = hull

    # tri par y puis x -> TL, TR en haut ; BL, BR en bas
    idx = np.argsort(pts[:,1], kind="mergesort")
    top = pts[idx[:2]][np.argsort(pts[idx[:2],0])]
    bottom = pts[idx[2:]][np.argsort(pts[idx[2:],0])]
    TL, TR = top[0], top[1]
    BL, BR = bottom[0], bottom[1]
    return np.array([TL, TR, BR, BL], dtype=np.float32)

def homographie(frame: np.ndarray):
    global _PLATE_SIZE_PIX

    frame_u = repere_distorsion(frame)

    centers = detection_centre_aruco(frame_u)
    if len(centers) < 4:
        raise RuntimeError(f"Seulement {len(centers)} ArUco détecté(s). 4 requis.")

    pts_img_all = np.array(list(centers.values()), dtype=np.float32)
    pts_img = ordre_aruco(pts_img_all)

    # Taille observée en pixels
    hull = cv2.convexHull(pts_img.reshape(-1,1,2)).astype(np.int32)
    x, y, w, h = cv2.boundingRect(hull)
    _PLATE_SIZE_PIX = (h, w)

    # orientation
    mm_w, mm_h = float(Plaque_largeur_mm), float(Plaque_longueur_mm)
    px_w, px_h = float(w), float(h)

    #échange de coté si mauvais coté
    swap_axes = ( (mm_w >= mm_h) != (px_w >= px_h) )
    if swap_axes:
        mm_w, mm_h = mm_h, mm_w
        print("↔️  Auto-orientation: axes mm échangés pour aligner grand côté mm avec grand côté pixels.")

    print(f"[Aspect] mm: {mm_w:.1f} x {mm_h:.1f}  |  px : {px_w:.0f} x {px_h:.0f} "
          f"(ratio mm={mm_w/mm_h:.3f} / px={px_w/px_h:.3f})")

    # Points cibles en mm dans frame rbot
    pts_mm = np.array([[0.0,   0.0],
                       [mm_w,  0.0],
                       [mm_w,  mm_h],
                       [0.0,   mm_h]], dtype=np.float32)

    # Homographie pixels -> mm
    H, ok = cv2.findHomography(pts_img, pts_mm, method=0)
    if H is None:
        raise RuntimeError("Échec du calcul de l'homographie.")

    ROI = cv2.convexHull(pts_img.reshape(-1,1,2)).astype(np.int32)

    return H, ROI

def homographie_du_repere(frame: np.ndarray):
    global _H, _ROI_IMG
    H, ROI = homographie(frame)
    _H = H
    _ROI_IMG = ROI
    return H, ROI

def convert_coord(x_pixel: float, y_pixel: float) -> Tuple[float, float]:
    if _H is None:
        raise RuntimeError("Homographie non initialisée. Appelle homographie_du_repere(frame) d'abord.")
    pt = np.array([[[float(x_pixel), float(y_pixel)]]], dtype=np.float32)
    mm = cv2.perspectiveTransform(pt, _H)
    return float(mm[0,0,0]), float(mm[0,0,1])

def interieur_roi(u: float, v: float) -> bool:
    if _ROI_IMG is None:
        return True
    return cv2.pointPolygonTest(_ROI_IMG, (float(u), float(v)), False) >= 0

# debug
def fenetre_debug(frame: np.ndarray, circles: List[Tuple[int,int,int]] = None,
               color_roi=(255,0,0), color_circle=(0,255,0)) -> np.ndarray:
    out = frame.copy()
    if _ROI_IMG is not None:
        cv2.polylines(out, [_ROI_IMG], True, color_roi, 2)
    if circles:
        for (u,v,r) in circles:
            cv2.circle(out, (int(u),int(v)), int(r), color_circle, 2)
            cv2.circle(out, (int(u),int(v)), 2, color_circle, -1)
    return out

def taille_plateau_px() -> Optional[Tuple[int,int]]:
    return _PLATE_SIZE_PIX
