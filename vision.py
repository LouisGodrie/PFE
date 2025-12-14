#==============================================================================
#
#
#                                vision.py
#
#
#==============================================================================

# Import Bibliothèques
import cv2
import numpy as np
import time


# Import sous-programmes
from prise_photo import prise_photo
from detect_circles import detect_circles
from detect_square import detect_squares
from detect_triangles import detect_triangles

from convert_coord import (
    convert_coord,
    homographie_du_repere,
    interieur_roi,
    taille_plateau_px,
    fenetre_debug,
    repere_distorsion,
    calibration_camera,
    distorsion_manuel,
)

# En cours
try:
    from convert_coord import get_last_h_stats
except Exception:
    def get_last_h_stats(): return {}


from interface_graph import creation_interface
from send_PR import send_PR
from send_D import send_DI
from get_R import getR22
from send_R import (
    send_integer_R90,  # R[90] = nombre total de pièces
    send_integer_R91,  # R[91] = type de pièce en cours
    send_integer_R92,  # R[92] = flag
)

# Masquage des Aruco post vision
from masque_aruco import construction_masque_aruco

#==============================Paramétrages===================================
SHOW_DEBUG    = True
TIMEOUT_R22   = 30.0
SWAP_XY_TO_ROBOT = True

USE_FULL_CALIB = False
ENABLE_UNDISTORT = True   # True = Image redressé selon MANUAL_K
MANUAL_K         = -0.15  # -0.15 --> Validé

# Calibration en cours (Non approuvé)
K = np.array([[1200.0, 0.0, 640.0],
              [0.0, 1200.0, 360.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)
D = np.array([-0.12, 0.03, 0.0, 0.0, 0.0], dtype=np.float32)

# DImension du plateau pour calibration Aruco
PLATE_W_MM = 260.0
PLATE_H_MM = 240.0

# Activation masquage Aruco + Dimension du masque
ENABLE_GLOBAL_ARUCO_MASK = True
GLOBAL_ARUCO_MARGIN_PX   = 12

# Paramétrage de la fenêtre de l'interface graphique
WINDOW_NAME = "ROI / Detections formes / Interface graphique"
UI_SCALE = 1.00
UI_MAX_W = 1600
UI_MAX_H = 900

#=========================Affichage interface graphique========================
def taille_interface(img: np.ndarray, scale: float = 1.0, max_w=None, max_h=None) -> np.ndarray:

    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    if max_w is not None and new_w > max_w:
        ratio = max_w / float(new_w)
        new_w = max_w
        new_h = int(new_h * ratio)
    if max_h is not None and new_h > max_h:
        ratio = max_h / float(new_h)
        new_h = max_h
        new_w = int(new_w * ratio)

    if new_w <= 0 or new_h <= 0:
        return img
    if new_w == w and new_h == h:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def interface_centre(win_name: str, img: np.ndarray, scale: float = 1.0, max_w=None, max_h=None):

    disp = taille_interface(img, scale=scale, max_w=max_w, max_h=max_h)

    # Fenêtre redimensionnable pour qu'OpenCV puisse appliquer move/resize proprement
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    h, w = disp.shape[:2]
    cv2.resizeWindow(win_name, w, h)

    # Récupère la résolution écran
    try:
        import ctypes
        user32 = ctypes.windll.user32
        sw = int(user32.GetSystemMetrics(0))
        sh = int(user32.GetSystemMetrics(1))
    except Exception:
        sw, sh = 1920, 1080

    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 2)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, disp)

#=======================Dessin détection des formes============================

def dessin_contours(base_bgr, x0, y0, circles, squares=None, triangles=None):
    vis = base_bgr.copy()
    # Cercles
    if circles and circles.get("cercles"):
        for c in circles["cercles"]:
            u = x0 + int(c["x"])
            v = y0 + int(c["y"])
            r = int(c.get("r", 15))
            cv2.circle(vis, (u, v), r, (0, 255, 0), 2)
            cv2.circle(vis, (u, v), 2, (0, 255, 0), -1)
            cv2.putText(vis, "Cercle", (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Carrés
    if squares and squares.get("squares"):
        for s in squares["squares"]:
            cx, cy = x0 + int(s["x"]), y0 + int(s["y"])
            if "approx" in s:
                cnt = np.asarray(s["approx"], dtype=np.int32)
                cnt[:,0,0] += x0
                cnt[:,0,1] += y0
                cv2.polylines(vis, [cnt], True, (255, 0, 255), 2)
            cv2.putText(vis, "Carre", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    # Triangles
    if triangles and triangles.get("triangles"):
        for t in triangles["triangles"]:
            cx, cy = x0 + int(t["x"]), y0 + int(t["y"])
            if "approx" in t:
                cnt = np.asarray(t["approx"], dtype=np.int32)
                cnt[:,0,0] += x0
                cnt[:,0,1] += y0
                cv2.polylines(vis, [cnt], True, (255, 255, 0), 2)
            cv2.putText(vis, "Triangle", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return vis

def lecture_R22():
    try:
        return int(getR22())
    except Exception:
        return 0




#====================================Cycle=====================================

def main():
    # Prise d'une photo de la caméra
    full = prise_photo()
    if full is None:
        print("Échec de la capture de l'image.")
        send_DI(False)
        return

    # Application de la distorsion si activée
    if ENABLE_UNDISTORT:
        if USE_FULL_CALIB: # Option désactivé (A garder)
            calibration_camera(K, D)
        else:
            distorsion_manuel(MANUAL_K)
        full = repere_distorsion(full)

    image = full

    # Homographie via Aruco
    homographie_du_repere(image)
    print("Homographie (pixels -> mm) initialisée.")

    plate_pix = taille_plateau_px()  # (h,w) du plateau en pixels
    if plate_pix:
        print(f"Taille en pixels du plateau : {plate_pix}")

    # Activation masque 
    exclusion_mask = construction_masque_aruco(image, GLOBAL_ARUCO_MARGIN_PX) if ENABLE_GLOBAL_ARUCO_MASK else None

    # Détections des formes
    x0 = y0 = 0
    circles = detect_circles(image, exclusion_mask=exclusion_mask)
    nC = len(circles.get("cercles", [])) if circles else 0
    print(f"Nombre de cercles après filtrage : {nC}")

    try:
        squares = detect_squares(image, exclusion_mask=exclusion_mask); nS = len(squares.get("squares", []))
        print(f"Nombre de carres après filtrage : {nS}")
    except Exception:
        squares, nS = None, 0

    try:
        triangles = detect_triangles(image, exclusion_mask=exclusion_mask); nT = len(triangles.get("triangles", []))
        print(f"Nombre de triangles après filtrage : {nT}")
    except Exception:
        triangles, nT = None, 0

    # Dessin des formes
    vis = dessin_contours(image, x0, y0, circles, squares, triangles)
    vis = fenetre_debug(vis)  # ROI

    # 8) Stats pour le panneau à droite
    px_per_mm = None
    if plate_pix:
        h_px, w_px = plate_pix
        px_per_mm = 0.5 * ((w_px / PLATE_W_MM) + (h_px / PLATE_H_MM))

    hstats = get_last_h_stats() or {}
    rms_mm = hstats.get("rms_mm")
    inliers = hstats.get("inliers")
    total_points = hstats.get("total")

    stats = {
        "camera_ok": True,
        "robot_ok": True,  # tu peux mettre False si un ping Modbus échoue
        "n_circles": nC,
        "n_squares": nS,
        "n_triangles": nT,
        "n_total": nC + nS + nT,
        "px_per_mm": px_per_mm,
        "roi_pixels": plate_pix,
        "rms_mm": rms_mm,
        "inliers": inliers,
        "total_points": total_points,
        "last_pr": None,
        "last_label": None,

    }

    # Titre des écrans de détection
    thumbs = []
    if isinstance(circles, dict) and "debug" in circles and "gray_blurred" in circles["debug"]:
        thumbs.append(("Cercles: gray+blur", circles["debug"]["gray_blurred"]))
    if isinstance(squares, dict) and "debug" in squares:
        if "bin" in squares["debug"]:
            thumbs.append(("Carres: Binaire", squares["debug"]["bin"]))
        if "mask" in squares["debug"]:
            thumbs.append(("Masque ArUco", squares["debug"]["mask"]))
    if isinstance(triangles, dict) and "debug" in triangles and "bin" in triangles["debug"]:
        thumbs.append(("Triangles: Binaire", triangles["debug"]["bin"]))

    dash = creation_interface(vis, stats, thumbs=thumbs)
    if SHOW_DEBUG:
        interface_centre(WINDOW_NAME, dash, scale=UI_SCALE, max_w=UI_MAX_W, max_h=UI_MAX_H)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Touche échap
            send_DI(False)
            cv2.destroyAllWindows()
            return

    # Mise en place des coordonnees resepectives des formes
    pieces = []
    
    # Cercle = label 1
    for c in circles.get("cercles", []):
        u, v = x0 + int(c["x"]), y0 + int(c["y"])
        if not interieur_roi(u, v):
            continue
        X, Y = convert_coord(u, v)
        pieces.append({"type": "Cercle", "label": 1, "X": X, "Y": Y})

    # Carrés = label 2
    if squares:
        for s in squares.get("squares", []):
            u, v = x0 + int(s["x"]), y0 + int(s["y"])
            if not interieur_roi(u, v):
                continue
            X, Y = convert_coord(u, v)
            pieces.append({"type": "Carre", "label": 2, "X": X, "Y": Y})

    # Triangles = label 3
    if triangles:
        for t in triangles.get("triangles", []):
            u, v = x0 + int(t["x"]), y0 + int(t["y"])
            if not interieur_roi(u, v):
                continue
            X, Y = convert_coord(u, v)
            pieces.append({"type": "Triangle", "label": 3, "X": X, "Y": Y})

    if not pieces:
        print("Aucune pièce détectée.")
        send_DI(False)
        if SHOW_DEBUG:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Contrôle du FLag robot et démarrage du programme robot 
    send_DI(True)
    send_integer_R90(len(pieces))
    print(f"R[90] <= {len(pieces)} (nombre total de pièces)")

    for idx, piece in enumerate(pieces, start=1):
        X, Y = piece["X"], piece["Y"]
        label = piece["label"]

        # Problème de repère côté PC, inversion du repère cartésien
        if SWAP_XY_TO_ROBOT:
            send_PR(Y, X)
            pr_log = f"PR=({Y:.3f}, {X:.3f}) [XY inversés]"
            stats["last_pr"] = (Y, X)
        else:
            send_PR(X, Y)
            pr_log = f"PR=({X:.3f}, {Y:.3f})"
            stats["last_pr"] = (X, Y)

        stats["last_label"] = label

        send_integer_R91(label)   # type
        print(f"[{idx}/{len(pieces)}] Type={label}. {pr_log}")

        send_integer_R92(1)       # handshake = données prêtes
        print("R[92] =1 --> Attente remise à 0")

        start = time.time()
        while True:
            try:
                r92 = int(getR22())
            except Exception:
                r92 = 0
            if r92 == 0:
                print("R[92] = 0 --> Robot prêt")
                break
            if time.time() - start > TIMEOUT_R22:
                print("Timeout ! Robot en repli")
                break
            time.sleep(0.1)

        # Actualisation
        if SHOW_DEBUG:
            dash = creation_interface(vis, stats, thumbs=thumbs)
            interface_centre(WINDOW_NAME, dash, scale=UI_SCALE, max_w=UI_MAX_W, max_h=UI_MAX_H)
            cv2.waitKey(1)

    print("Toutes les pièces ont été déposées")
    send_DI(False)

    if SHOW_DEBUG:
        print("Échap ESC pour fermer les fenêtres")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()