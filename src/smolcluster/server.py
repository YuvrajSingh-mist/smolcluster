import socket

HOST_IP = socket.gethostbyname(socket.gethostname())
PORT = 65432  # Port to listen on

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((HOST_IP, PORT))

# Listen for incoming connections
server_socket.listen(5)
print(f"Server listening on {HOST_IP}:{PORT}")

# Accept connections
while True:
    client_socket, client_address = server_socket.accept()
    print(f"Accepted connection from {client_address}")
    # Handle the connection (you can add more logic here)
    client_socket.close()

