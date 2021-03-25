import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
# 这意味着，无论你有一个等于padding_idx的项，该索引处的嵌入层的输出都将是0。