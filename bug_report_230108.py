import oneflow as flow
import numpy as np
import oneflow.nn as nn
from oneflow.one_embedding import make_persistent_table_writer

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = 128
        
        self.fc0_w_dict = dict()
        self.fc0_b_dict = dict()
        self.fc1_w_dict = dict()
        self.fc1_b_dict = dict()
        
        for i in range(self.layers):
            self.fc0_w_dict[i] = flow.randn((768, 3072), dtype=flow.float32)
            self.fc0_b_dict[i] = flow.randn((3072), dtype=flow.float32)
            self.fc1_w_dict[i] = flow.randn((3072, 768), dtype=flow.float32)
            self.fc1_b_dict[i] = flow.randn((768), dtype=flow.float32)

    def forward(self, x):
        for i in range(self.layers):
            # 需求实现：显存不足，forward时临时将权重从CPU Load到GPU
            fc0_w = self.fc0_w_dict[i].cuda()
            fc0_b = self.fc0_b_dict[i].cuda()
            fc1_w = self.fc1_w_dict[i].cuda()
            fc1_b = self.fc1_b_dict[i].cuda()
            x = flow.einsum('...nd,...dh->...nh', x, fc0_w) + fc0_b
            x = flow.einsum('...nh,...hd->...nd', x, fc1_w) + fc1_b
        return x



model= Model()
model.to("cuda")


class Graph(nn.Graph):
    def __init__(self, ):
        super().__init__()
        self.model = model
        # 打开了省内存选项
        self.config.enable_compress_memory(True)

    def build(self, x):
        y = self.model(x)
        return y


graph = Graph()
    
with flow.no_grad():
    for _ in range(2):
        x = flow.rand((1, 768), dtype=flow.float32).to("cuda")
        # 动态图测试
        # y = model(x)
        # 静态图测试
        y = graph(x)

import pdb
pdb.set_trace()
