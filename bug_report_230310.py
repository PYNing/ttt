import time
import oneflow.nn as nn
import oneflow as flow


def set_auto_offload_hooks(model):
    def auto_load_hook(module, input):
        before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        for param in module.parameters():
            if param.is_offloaded:
                param.load()
        after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print(f"auto load\tbefore: {before_used}\tafter: {after_used}")
        return None

    def auto_offload_hook(module, input, output):
        before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        for param in module.parameters():
            if not param.is_offloaded:
                param.offload()
        after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print(f"auto offload\tbefore: {before_used}\tafter: {after_used}")
        return None

    model.register_forward_pre_hook(auto_load_hook)
    model.register_forward_hook(auto_offload_hook)
    return model


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


model = Model()
model = set_auto_offload_hooks(model)
model.cuda()
model.eval()

BZ = 128
dataset = [flow.rand((BZ, 768), dtype=flow.float32) for _ in range(4)]

print("RUN")
st = time.time()
with flow.no_grad():
    for idx, x in enumerate(dataset):
        print(f"iter {idx} begin")
        x = x.cuda()

        y = model(x)

        flow._oneflow_internal.eager.Sync()
        print(f"iter {idx} end")

et = time.time()
print("throupout: %f sample/s" % (BZ * len(dataset) / (et - st)))
