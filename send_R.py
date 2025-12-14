#==============================================================================
#
#
#                           send_R.py
#
#
#==============================================================================

from pymodbus.client import ModbusTcpClient

def send_integer_R90(value):
    # Config modbus
    client = ModbusTcpClient('192.168.2.10', port=502)

    # Connexion au robot
    if client.connect():
        print("Connecté au robot via Modbus.")

        # conversion en 32 bits
        registres = [value & 0xFFFF, (value >> 16) & 0xFFFF]

        # Écriture registres modbus
        client.write_registers(89, registres)
        print(f"Nombre entier envoyé : {value}")
    else:
        print("Connexion au robot échouée.")

    # Déconnexion
    client.close()

def send_integer_R91(value):
    client = ModbusTcpClient('192.168.2.10', port=502)

    # Connexion au robot
    if client.connect():
        print("Connecté au robot via Modbus.")
        registres = [value & 0xFFFF, (value >> 16) & 0xFFFF]

        client.write_registers(90, registres)
        print(f"Nombre entier envoyé : {value}")
    else:
        print("Connexion au robot échouée.")

    # Déconnexion
    client.close()


def send_integer_R92(value):
    client = ModbusTcpClient('192.168.2.10', port=502)

    if client.connect():
        print("Connecté au robot via Modbus.")
        registres = [value & 0xFFFF, (value >> 16) & 0xFFFF]
        client.write_registers(91, registres)
        print(f"Nombre entier envoyé : {value}")
    else:
        print("Connexion au robot échouée.")

    # Déconnexion
    client.close()

send_integer_R92(1)