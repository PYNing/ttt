import oneflow as flow
import numpy as np
import oneflow.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(768, 3072)
        self.fc1 = nn.Linear(3072, 768)
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = 12
        self.layers = nn.ModuleList([MLP() for _ in range(self.layers)])
        

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


model= Model()
model.to("cuda")

class Graph(nn.Graph):
    def __init__(self, ):
        super().__init__()
        self.model = model

    def build(self, x):
        y = self.model(x)
        return y


graph = Graph()
# graph.debug(3)

with flow.no_grad():
    for _ in range(100):
        x = flow.rand((1, 768), dtype=flow.float32).to("cuda")
        y = graph(x)

import pdb
pdb.set_trace()
