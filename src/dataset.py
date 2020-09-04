from torch.utils.data import Dataset
import pandas as pd
from src.utils import config
from src.utils.tools import clean_symbols, query_cut
from tqdm import tqdm
import torch

tqdm.pandas()


class NewsDataset(Dataset):
    """数据集的创建"""

    def __init__(self, path, dictionary=None, tokenizer=None, word=False, debug=False):
        super(NewsDataset, self).__init__()
        self.word = word
        self.data = pd.read_csv(path, sep='\t').dropna()
        if debug:
            self.data = self.data.sample(n=1000, replace=True)
        self.data["sentence"] = self.data['title']
        # self.data["sentence"] = self.data['title'] + self.data['content']
        # self.data["sentence"] = self.data['content']
        self.data['clean_sentence'] = self.data['sentence'].progress_apply(clean_symbols)
        self.data["cut_sentence"] = self.data['clean_sentence'].progress_apply(query_cut)
        # 标签映射到id
        self.data['category_id'] = self.data['label'].progress_apply(lambda x: x.strip()).map(config.label2id)
        # char粒度
        if not self.word:
            self.data['cut_sentence'] = self.data['cut_sentence'].progress_apply(lambda x: ' '.join(''.join(x)))
        self.dictionary = dictionary

    def __getitem__(self, i):
        data = self.data.iloc[i]
        text = data['cut_sentence']  # text数据
        labels = int(data['category_id'])
        if not self.word:
            text = text.split(' ')
        input_ids = [self.dictionary.indexer(t) for t in text]  # 获取每个token的id
        output = {
            "token_ids": input_ids,
            "labels": labels
        }
        return output

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和label
    """

    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])  # batch中样本的最大的长度
    labels = torch.tensor([data["labels"] for data in batch])

    token_ids_padded = padding(token_ids, max_length)  # 填充每个batch的sample
    return token_ids_padded, labels
