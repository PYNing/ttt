import oneflow as flow
import numpy as np
import oneflow.nn as nn
from oneflow.one_embedding import make_persistent_table_writer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        CACHE_BUDGET_MB = 512
        # 这里的数值含义请看forward()
        self.rows_per_layer = 3072 + 4 + 3072 + 1
        self.embedding_size = 768
        self.layers = 12
        table_size_array = [self.layers * self.rows_per_layer]
        vocab_size = sum(table_size_array)

        scales = np.sqrt(1 / np.array(table_size_array))
        tables = [
            flow.one_embedding.make_table_options(
                flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
            )
            for scale in scales
        ]
        store_options = flow.one_embedding.make_cached_ssd_store_options(
            cache_budget_mb=CACHE_BUDGET_MB, persistent_path="gpt2_ssd", capacity=vocab_size,
        )

        embedding = flow.one_embedding.MultiTableEmbedding(
            name="my_embedding",
            embedding_dim=self.embedding_size,
            dtype=flow.float,
            key_type=flow.int32,
            tables=tables,
            store_options=store_options,
        )

        embedding.save_snapshot("gpt2")
        tables = ["gpt2_ssd/0-1"]  # 推理场景，仅单卡
        # 给Embbedding填入有意义的数值（如GPT-2的权重）
        with make_persistent_table_writer(tables, "gpt2", flow.int32, flow.float, self.embedding_size) as writer:
            keys = np.arange(self.layers * self.rows_per_layer).astype(np.int32)
            values = np.random.randn(self.layers * self.rows_per_layer * self.embedding_size).reshape(
                (self.layers * self.rows_per_layer, self.embedding_size)).astype(np.float32)
            writer.write(keys, values)
        embedding.load_snapshot("gpt2")

        self.embedding = embedding

    def forward(self, x):
        # 这里模仿了GPT-2的前向
        for layer_i in range(self.layers):
            # 每个layer有2个fc
            # 第0个fc的weight的维度是[768, 3072]
            # 第0个fc的bias的维度是[3072]
            # 第1个fc的weight的维度是[3072, 768]
            # 第1个fc的bias的维度是[768]
            # 取这些维度的最大公约数[768]作为Embedding每个数据项的维度
            # 每次取连续的6149行（3072 + 4 + 3072 + 1），即可将一层的权重都取出来
            look_up_ids_start = layer_i * self.rows_per_layer
            look_up_ids_end = (layer_i + 1) * self.rows_per_layer
            ids = np.arange(start=look_up_ids_start, stop=look_up_ids_end, dtype=np.int32)
            ids_tensor = flow.tensor(ids, requires_grad=False).to("cuda")
            weight_bias = self.embedding(ids_tensor)

            print(f"layer{layer_i}")
            print("lookup tensor shape:")
            print(ids_tensor.shape)
            print("embbeding tensor shape")
            print(weight_bias.shape)
            print("*" * 50)

            # slice、reshape，还原为4个部分的权重
            fc0_w = weight_bias[0: 3072, :].reshape(768, 3072)
            fc0_b = weight_bias[3072: 3076, :].reshape(-1)
            fc1_w = weight_bias[3076:6148, :].reshape(3072, 768)
            fc1_b = weight_bias[6148:6149, :].reshape(-1)

            # FFN逻辑
            x = flow.einsum('...nd,...dh->...nh', x, fc0_w) + fc0_b
            x = flow.einsum('...nh,...hd->...nd', x, fc1_w) + fc1_b
        return x


model = Model()
model.to("cuda")


class Graph(nn.Graph):
    def __init__(self, ):
        super().__init__()
        self.model = model

    def build(self, x):
        y = self.model(x)
        return y


graph = Graph()

with flow.no_grad():
    x = flow.rand((1, 768), dtype=flow.float32).to("cuda")
    y = graph(x)
