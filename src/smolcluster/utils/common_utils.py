import socket
import pickle, struct

def send_message(sock: socket.SocketType, message: dict):
    
    data = pickle.dumps(message)
    sock.sendall(struct.pack('>I', len(data)) + data)
    
        
def receive_message(sock: socket.SocketType) -> dict:
    
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = b""
    while True:
        chunk = sock.recv(msglen)
        data += chunk
        if len(data) < msglen:
            msglen -= len(chunk)
        else:
            break
    return pickle.loads(data)
    
