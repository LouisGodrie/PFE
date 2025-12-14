#==============================================================================
#
#
#                           prise_photo.py
#
#
#==============================================================================

# Import des bilbiothèques
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension
import cv2
import numpy as np

def prise_photo():
# Initialisation de la librarie
    ids_peak.Library.Initialize()

# Création d'un device manager
    device_manager = ids_peak.DeviceManager.Instance()

    try:
# Mise à jour du device manager
        device_manager.Update()

# Fin du programme si aucun 
        if device_manager.Devices().empty():
            print("No device found. Exiting Program.")
            return

# Ouverture de la première caméra détectée avec accès "Control"
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)

# Accès au nodemap du périphérique
        nodemap_remote_device = device.RemoteDevice().NodeMaps()[0]

# Affiche le modèle de la caméra
        print("Opening camera model: " + nodemap_remote_device.FindNode("DeviceModelName").Value())

# Réglage du temps d'exposition 
        nodemap_remote_device.FindNode("ExposureTime").SetValue(14000)
        print("Exposure Time: ", nodemap_remote_device.FindNode("ExposureTime").Value())

# Ouverture du flux de données (datastream) pour recevoir les images
        datastreams = device.DataStreams()
        datastream = datastreams[0].OpenDataStream()

# Taille du payload (taille de l'image) pour allouer les buffers correctement
        payload_size = nodemap_remote_device.FindNode("PayloadSize").Value()

        for i in range(datastream.NumBuffersAnnouncedMinRequired() + 1):
            buffer = datastream.AllocAndAnnounceBuffer(payload_size)
            datastream.QueueBuffer(buffer)

# Démarrage de l'acquisition de l'image
        datastream.StartAcquisition()
        nodemap_remote_device.FindNode("AcquisitionStart").Execute()

# Attente d'un buffer de 5000 ms terminé 
        buffer = datastream.WaitForFinishedBuffer(5000)

# Conversion du buffer brut en image IDS peak IPL
        ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)

# Conversion au format BGR8 (compatible avec OpenCV)
        converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_BGR8)

# Récupération sous forme de tableau NumPy
        picture = converted_ipl_image.get_numpy_3D()

# Redimensionnement pour affichage
        picture = cv2.resize(picture, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Captured Photo", picture)
        cv2.waitKey(0)  # attente d'une touche pour fermer la fenêtre

# Sauvegarde de l'image sur disque
        photo_filename = "captured_photo.png"
        cv2.imwrite(photo_filename, picture)
        print(f"Photo captured and saved as {photo_filename}")
        return picture

    except Exception as e:
        print("Exception: " + str(e))
    finally:
# Arrêt du flux
        datastream.KillWait()
        datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        ids_peak.Library.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    prise_photo()
