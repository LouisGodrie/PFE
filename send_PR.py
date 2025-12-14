#==============================================================================
#
#
#                           send_PR.py
#
#
#==============================================================================

from pymodbus.client import ModbusTcpClient

def send_PR(x, y):
# Configuration Modbus tcp ip
    client = ModbusTcpClient('192.168.2.10', port=502)

# Connexion au robot
    if client.connect():
        print("Connecté au robot via Modbus.")

        # Valeurs à écrire dans PR[1]
        valeurs = [x, y, -38, -180, 0, 0]



        # Multiplication pour gérer les décimales
        multiply_factor = 1000
        valeurs_converties = [int(v * multiply_factor) for v in valeurs]

        # Préparation des registres
        registres = []
        for valeur in valeurs_converties:
            registres.append(valeur & 0xFFFF)  # 16 bits
            registres.append((valeur >> 16) & 0xFFFF)

        # Écriture registre modbus
        client.write_registers(17, registres)
        print(f"Coordonnées PR[1] mises à jour avec : {valeurs}")
    else:
        print("Connexion au robot échouée.")

    # Déconnexion
    client.close()

