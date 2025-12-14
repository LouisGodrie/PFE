#==============================================================================
#
#
#                           send_D.py
#
#
#==============================================================================

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# Configuration de connexion
robot_ip = '192.168.2.10'
modbus_port = 502

# Adresse pour DI[1]
coil_address = 1

def send_DI(value):
    try:
# Création d'un client Modbus
        with ModbusTcpClient(robot_ip, port=modbus_port) as client:
# Vérification de la connexion
            if not client.connect():
                print("Erreur : impossible de se connecter au serveur Modbus.")
                return False

# Écriture de la valeur sur la sortie
            response = client.write_coil(coil_address, value)
            if response.isError():
                print(f"Erreur lors de l'écriture sur DI[1] à l'adresse {coil_address}.")
                return False
            else:
                print(f"DI[1] mise à {'ON' if value else 'OFF'} à l'adresse {coil_address}.")
                return True
    except ModbusException as e:
        print(f"Exception Modbus : {e}")
        return False
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return False

# Test de la fonction
if __name__ == "__main__":
    success = send_DI(False)
    print(f"Résultat de l'opération : {'Succès' if success else 'Échec'}")
