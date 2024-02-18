import socket


class RPCCommunication:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP

    def send_controls(self, wheel, throttle):
        UDP_IP = "10.3.23.52"
        UDP_PORT = 90
        MESSAGE = f"""{{
            "jsonrpc": "2.0",
            "method": "set_values",
            "params": [{wheel}, {throttle}]
        }}"""

        self.sock.sendto(MESSAGE.encode(), (UDP_IP, UDP_PORT))
