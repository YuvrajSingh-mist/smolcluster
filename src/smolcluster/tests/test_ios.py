import socket

import numpy as np

ipad_ip = "172.70.1.232"  # replace with real IP

x = np.random.randn(1, 1, 768).astype(np.float32)

s = socket.create_connection((ipad_ip, 8000))
s.sendall(x.tobytes())

data = b""
while len(data) < x.nbytes:
    data += s.recv(4096)

y = np.frombuffer(data, dtype=np.float32).reshape(1, 1, 768)
print("Returned shape:", y.shape)
