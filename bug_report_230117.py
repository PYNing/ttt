import oneflow as flow
import oneflow.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, x):
        x1 = self.linear(x)
        x2 = x * 2
        return x1 + x2

model = Model().cuda()

class Graph(nn.Graph):
    def __init__(self) -> None:
        super().__init__()
        self.model = model

    def build(self, x):
        return self.model(x)

graph = Graph() 

x = flow.randn([1, 128], dtype=flow.float32).cuda()

y = graph(x)
y = graph(x)
print(graph)
