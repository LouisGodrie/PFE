# PFE
Prérequis pour utiliser le code :
 
-Avoir python, code testé avec python 3.12
-Téléchargez les drivers de la caméra sur ce site : https://pypi.org/project/ids_peak
( prendre version qui correspond à votre version de python et votre OS

Décompresser le dossier puis mettre les fichiers whl dans le repertoire courant du projet.

Lorsque c'est tapez la commande suivante :
 python -m pip install --no-index --find-links=. ids_peak

Ca cherche tous les fichiers whl dans le repertoire courant puis installe les dépendances hors lignes.


Pour vérifier l'installation : 

python -c "import ids_peak; print('ids_peak est installé correctement')"


Pour se connecter en ethernet au robot (pour envoyer signal et coordonnées) il faut une fois branchée se définir une adresse IP sur le réseau
par exemple "192.168.1.50" et mettre 255.255.255.0 en masque de sous réseau


module a installer : pymodbus, opencv-python
