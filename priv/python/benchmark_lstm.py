#!/usr/bin/env python3
import torch
batch,seq,hidden=32,60,512
x=torch.randn(batch,seq,hidden,device='cuda')
lstm=torch.nn.LSTM(hidden,256,batch_first=True).cuda()
[lstm(x) for _ in range(10)]
torch.cuda.synchronize()
s=torch.cuda.Event(enable_timing=True)
e=torch.cuda.Event(enable_timing=True)
s.record()
[lstm(x) for _ in range(100)]
e.record()
torch.cuda.synchronize()
print(f'LSTM: {s.elapsed_time(e)/100:.3f} ms')
