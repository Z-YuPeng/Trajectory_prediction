import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False #不需要求梯度

        position = torch.arange(0, max_len).float().unsqueeze(1) # 增加一个维度，相当于在第几维增加一个括号！ max_len * 1
        # 维度从0开始，squeeze()正好反过来，减少维度，length（em..）为1的维度就被删除了
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()      # d_model/2
        # 这一步在干啥？ 先跳过吧，不求甚解，反正是计算位置编码，dmodel是什么意思？不知道
        pe[:, 0::2] = torch.sin(position * div_term)   #这个乘法算下来之后应该是  max_len * d_model/2
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe : max_len * d_model
        pe = pe.unsqueeze(0) # 又在第0维增加维度，1* max_len * d_model
        self.register_buffer('pe', pe)  #  buffer对象  buffer和parameter的区别？https://zhuanlan.zhihu.com/p/130885539

    def forward(self, x):
        return self.pe[:, :x.size(1)]
