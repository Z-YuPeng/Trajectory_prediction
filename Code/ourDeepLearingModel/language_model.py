import torch.nn as nn

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_lm(x)




class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        # self.linear = nn.Linear(hidden, vocab_size)
        # fc1 = nn.Linear(hidden, 256)
        self.linear = nn.Linear(hidden, vocab_size)
        # fc2= nn.Linear(256,256)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     fc1,
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     fc2,
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
