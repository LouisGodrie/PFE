#==============================================================================
#
#
#                           get_R.py
#
#
#==============================================================================


from pymodbus.client import ModbusTcpClient

def getR22():
    ROBOT_IP = "192.168.2.10"
    MODBUS_PORT = 502

    client = ModbusTcpClient(ROBOT_IP, port=MODBUS_PORT)

    try:
        if client.connect():
            # print("Connecté au robot FANUC via MODBUS/TCP")
            registre_address = 91  # R[22] en base 0 !!!
            result = client.read_holding_registers(address=registre_address, count=1)

            if (not result.isError()) and getattr(result, "registers", None):
                valeur = result.registers[0]
                return int(valeur)
            else:
                print("Erreur de lecture MODBUS ou aucun registre renvoyé")
                return 0
        else:
            print("Échec de la connexion au robot FANUC")
            return 0
    finally:
        client.close()
