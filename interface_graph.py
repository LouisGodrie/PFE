#==============================================================================
#
#
#                           interface_graph.py
#
#
#==============================================================================

# Import bibliothèques
import cv2
import numpy as np

# COuleur du fond de l'interface graphique
BG = (32, 32, 38)

# Voyant connexion robot + texte
def Voyant(panel, x, y, ok: bool, label: str):
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.circle(panel, (x, y), 8, color, -1)
    cv2.putText(panel, label, (x+18, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

# Création d'une colonne pour afficher diverses informations
def Colonne_affichage(panel, x, y, key: str, val: str):
    cv2.putText(panel, key, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(panel, val, (x+180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)




# A SUPPRIMER
def _bar(panel, x, y, w, h, ratio: float, label: str):
    ratio = max(0.0, min(1.0, float(ratio)))
    cv2.rectangle(panel, (x, y), (x+w, y+h), (70,70,70), 1)
    cv2.rectangle(panel, (x+1, y+1), (x+1+int((w-2)*ratio), y+h-1), (0,180,0), -1)
    cv2.putText(panel, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

def background(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# Création d'une vignette danbs l'intefarce personnalisable
def Vignette(img_bgr, W, H, title=None, pad_color=BG):
    """
    Place 'img_bgr' DANS WxH SANS déformation (respect aspect), puis pad pour obtenir EXACTEMENT WxH.
    """
    img = background(img_bgr)
    if img is None:
        return np.full((H, W, 3), pad_color, np.uint8)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((H, W, 3), pad_color, np.uint8)

    scale = min(W / float(w), H / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (H - new_h) // 2
    bottom = H - new_h - top
    left = (W - new_w) // 2
    right = W - new_w - left

    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)

    cv2.rectangle(canvas, (0,0), (W-1,H-1), (60,60,60), 1)
    if title:
        cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220,220,220), 2, cv2.LINE_AA)
    return canvas

def creation_interface(image_bgr: np.ndarray, stats: dict, panel_w: int = 360,
                       thumbs=None, scale_out: float = 1.0) -> np.ndarray:
    img = image_bgr
    H, W = img.shape[:2]

    # -------- Panneau d'infos
    panel = np.full((H, panel_w, 3), BG, dtype=np.uint8)
    y = 36
    cv2.putText(panel, "Vision  Dashboard", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    y += 34

    # LEDs état
    def Voyant(panel, x, y, ok: bool, label: str):
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.circle(panel, (x, y), 8, color, -1)
        cv2.putText(panel, label, (x+18, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

    Voyant(panel, 18, y, stats.get("camera_ok", False), "Camera")
    Voyant(panel, 140, y, stats.get("robot_ok", False),  "Robot")
    y += 28

    cv2.putText(panel, "Detections", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180,220,255), 2, cv2.LINE_AA); y += 26
    Colonne_affichage(panel, 16, y,  "Cercles:",   str(stats.get("n_circles", 0)));   y += 24
    Colonne_affichage(panel, 16, y,  "Carres:",    str(stats.get("n_squares", 0)));   y += 24
    Colonne_affichage(panel, 16, y,  "Triangles:", str(stats.get("n_triangles", 0))); y += 24
    cv2.line(panel, (16, y+6), (panel_w-16, y+6), (70,70,80), 1); y += 18
    Colonne_affichage(panel, 16, y,  "Total:",     str(stats.get("n_total", 0)));     y += 28

    cv2.putText(panel, "Calibration", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180,220,255), 2, cv2.LINE_AA); y += 26
    roi_hw = stats.get("roi_pixels")
    if roi_hw:
        Colonne_affichage(panel, 16, y, "ROI (px):", f"{roi_hw[1]}x{roi_hw[0]}"); y += 24
    pxmm = stats.get("px_par_mm")
    if pxmm:
        Colonne_affichage(panel, 16, y, "Echelle:", f"{pxmm:.3f} px/mm"); y += 24

    rms = stats.get("rms_mm")
    if rms is not None:
        conf = max(0.0, min(1.0, 1.0 - (rms/2.0)))
        _bar(panel, 16, y+6, panel_w-32, 14, conf, f"Confiance (RMS={rms:.2f} mm)")
        y += 38
    else:
        y += 10

    last_pr = stats.get("last_pr")
    if last_pr is not None:
        cv2.putText(panel, "Dernier envoi", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180,220,255), 2, cv2.LINE_AA); y += 26
        Colonne_affichage(panel, 16, y, "PR(X,Y):", f"{last_pr[0]:.2f}, {last_pr[1]:.2f}"); y += 24
        lbl = stats.get("last_label")
        if lbl:
            noms = {1:"Cercle", 2:"Carre", 3:"Triangle"}
            Colonne_affichage(panel, 16, y, "Type:", noms.get(lbl, str(lbl))); y += 26

    exp = stats.get("exposure_ms")
    if exp:
        Colonne_affichage(panel, 16, y, "Exposure:", f"{exp:.1f} ms"); y += 24

    # rangée du haut du dbug
    top_row = np.hstack([img, panel])
    top_w = top_row.shape[1]

    # rangée du bas du debug
    strip = None
    if thumbs:
        cells = []
        gap = 16
        gap_img = np.full((H, gap, 3), BG, np.uint8)

        for title, im in thumbs:
            cell = Vignette(im, W, H, title=title, pad_color=BG)
            cells.append(cell)
            cells.append(gap_img)
        if cells:
            cells = cells[:-1]  # retire le dernier gap
            strip = np.hstack(cells)

    if strip is None:
        canvas = top_row
    else:
        right_blank = np.full((H, panel_w, 3), BG, dtype=np.uint8)
        bottom_row = np.hstack([strip, right_blank])
        bottom_w = bottom_row.shape[1]

        # Harmonise les largeurs pour vstack
        if bottom_w > top_w:
            pad = np.full((top_row.shape[0], bottom_w - top_w, 3), BG, dtype=np.uint8)
            top_row = np.hstack([top_row, pad])
        elif top_w > bottom_w:
            pad = np.full((bottom_row.shape[0], top_w - bottom_w, 3), BG, dtype=np.uint8)
            bottom_row = np.hstack([bottom_row, pad])

        canvas = np.vstack([top_row, bottom_row])

    # Zoom de sortie
    scale_out = 1.0
    if abs(scale_out - 1.0) > 1e-3:
        canvas = cv2.resize(canvas, (int(canvas.shape[1]*scale_out), int(canvas.shape[0]*scale_out)),
                            interpolation=cv2.INTER_LINEAR)
    return canvas
