#==============================================================================
#
#
#                           detect_circles.py
#
#
#==============================================================================

import cv2
import numpy as np

def détection_ref(cropped_image, exclusion_mask=None):
    # applique masque si fourni
    if exclusion_mask is not None:
        img = cropped_image.copy()
        img[exclusion_mask > 0] = (0, 0, 0)
    else:
        img = cropped_image

# Conversion en niveaux de gris
    gray_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Flou gaussien pour réduire le bruit et stabiliser la détection des contours
    gray_blurred = cv2.GaussianBlur(gray_cropped, (5, 5), 0)
# Détection de cercles par transformée de Hough
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=20,
        param1=140,
        param2=10,
        minRadius=3,        # rayon minimum recherché
        maxRadius=15        # rayon maximum recherché
    )

    result = {}
    if circles is not None:
        circles = np.uint16(np.around(circles))
        closest_circle = min(circles[0], key=lambda c: c[0])
        if closest_circle is not None:
            x, y, r = closest_circle
            result["closest_x"] = int(x)
            result["closest_y"] = int(y)
            result["closest_r"] = int(r)
    else:
        print("Aucun cercle détecté.")

    print(len(result))
    return result


def detect_circles(cropped_image, exclusion_mask=None):
    # applique masque si fourni
    if exclusion_mask is not None:
        img = cropped_image.copy()
        img[exclusion_mask > 0] = (0, 0, 0)
    else:
        img = cropped_image

    gray_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray_cropped, (5, 5), 0)
# Détection de cercles par transformée de Hough
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=20,
        param1=140,
        param2=33,
        minRadius=3,
        maxRadius=30
    )

    image_with_circles = img.copy()
    result = {"cercles": [], "debug": {"gray_blurred": gray_blurred}}
# Conversion + arrondi des valeurs détectées
    if circles is not None:
        circles = np.uint16(np.around(circles))
# Filtre pour ne garder que les cercles dont le rayon >= 10
        circles = np.array([circle for circle in circles[0] if circle[2] >= 10])

        print(f"Nombre de cercles après filtrage : {len(circles)}")

        for x, y, r in circles:
            result["cercles"].append({"x": int(x), "y": int(y), "r": int(r)})
            cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 2)
            cv2.circle(image_with_circles, (x, y), 1, (0, 0, 255), -1)
    else:
        print("Aucun cercle détecté.")

    # affichage direct retiré pour rester cohérent avec l'UI globale
    return result
