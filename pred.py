from main import make_model,greedy_decode
import torch
import numpy as np
from torch.autograd import Variable

V = 11
model = make_model(V, V, N=2)
model.load_state_dict(torch.load("./models/transformer_model.pt"))
model.eval()

for _ in range(10):
    source = Variable(torch.from_numpy(np.random.randint(1,10,size=(1, 10))))
    source_mask = Variable(torch.ones(1, 1, 10))
    res = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print('原始数据：', source)
    print('目标数据：',res)
    print('---------------------------------------')
