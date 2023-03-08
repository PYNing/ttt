import time
import sys
import os

# os.environ["ONEFLOW_DEBUG"] = "1"
# os.environ["ONEFLOW_PYTHON_STACK_GETTER"] = "1"

import oneflow.nn as nn
import oneflow as flow

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layer = 4

        layer_list = list()

        for _ in range(self.n_layer):
            layer_list.append(nn.Linear(7680, 40960))
            layer_list.append(nn.Linear(40960, 7680))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


model = Model().cuda()
print(model)

BZ = 128
dataset = [flow.rand((BZ, 7680), dtype=flow.float32) for _ in range(128)]


print("RUN")
st = time.time()
# BUG 0: set_autocast_enabled fail
flow.set_autocast_enabled = True
print(f"flow.is_autocast_enabled={flow.is_autocast_enabled()}")
with flow.no_grad():
    
    # for idx, x in enumerate(dataset):
    #     print(f"iter={idx}")
    #     x = x.cuda()
    #     y = model(x)
    
    with flow.autocast("cuda"):
        for idx, x in enumerate(dataset):
            print(f"iter={idx}")
            x = x.cuda()
            y = model(x)
            
et = time.time()
# BUG 1: slower than that without flow.autocast("cuda")
print("throupout: %f sample/s" %  (BZ * len(dataset) / (et - st)))

# BUG 2: Crash when finish with "with flow.autocast("cuda")" enable
