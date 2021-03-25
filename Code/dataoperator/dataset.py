from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            # if on_memory: 这个就是选择把数据加载进内存还是不加载进内存
            prelines = [line[:-1].split(",")  # [:-1] 可以去掉最后一个换行符号 \n!!!!
                        for line in f]

            # if on_memory: 这个就是选择把数据加载进内存还是不加载进内存
            self.lines = []
            for line in prelines:
                if len(line) < seq_len:
                    self.lines.append(line)
                else:
                    for i in range(seq_len,len(line),seq_len//2):
                        self.lines.append(line[i-seq_len:i])
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):#返回一个句子
        t1 = self.getitem(item)
        t1_random, t1_label = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        # t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t1=t1_random
        t1_label=t1_label
        bert_input = t1[:self.seq_len]
        bert_label = t1_label[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
            # , segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        for i, token in enumerate(tokens):
            if i >0 and tokens[i-1] == token:
                continue
            prob = random.random()
            if prob < 0.25:
                # prob /= 0.25

                # 80% randomly change token to mask token
                # if prob < 0.8:
                tokens[i] = self.vocab.mask_index

                # # 10% randomly change token to random token
                # elif prob < 0.9:
                #     tokens[i] = random.randrange(len(self.vocab))
                #
                # # 10% randomly change token to current token
                # else:
                #     tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0) # 只要没mask，label都写0

        return tokens, output_label
    def getitem(self, index):
        t1 = self.get_corpus_line(index)
        return t1

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]
        else:
            print("进入了未实现的分支")
            exit()



