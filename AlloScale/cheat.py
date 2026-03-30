import torch
import time

N = 32768

while True:
    for id in range(2):
        a = torch.randn(N, N, device=f"cuda:{id}")
        b = torch.randn(N, N, device=f"cuda:{id}")
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    time.sleep(10)
