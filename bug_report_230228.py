import time
import oneflow.nn as nn
import oneflow as flow

def load_eager_model(model):
    for param in model.parameters():
        if param.data.is_offloaded():
            param.data.load()

def offload_eager_model(model):
    for param in model.parameters():
        if not param.data.is_offloaded():
            param.data.offload()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layer = 4

        layer_list = list()

        for _ in range(self.n_layer):
            layer_list.append(nn.Linear(768, 4096))
            layer_list.append(nn.Linear(4096, 768))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
    

model0 = Model().cuda()
model1 = Model().cuda()

BZ = 128
dataset = [flow.rand((BZ, 768), dtype=flow.float32) for _ in range(128)]


print("RUN")
st = time.time()
rank = flow.env.get_rank()
with flow.no_grad():
    for idx, x in enumerate(dataset):
        print(f"iter {idx} begin")
        x = x.cuda()

        load_eager_model(model0)
        y0 = model0(x)
        offload_eager_model(model0)

        load_eager_model(model1)
        y1 = model1(x)
        offload_eager_model(model1)

        y = (y0 + y1) / 2

        print(f"iter {idx} end")

et = time.time()
print("throupout: %f sample/s" %  (BZ * len(dataset) / (et - st)))
