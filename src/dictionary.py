from collections import Counter
from src.utils.tools import build_dict_dataset
from src.utils import config
import joblib


class Dictionary(object):
    """构建词典"""
    def __init__(self,
                 max_vocab_size=50000,
                 min_count=None,
                 start_end_tokens=False):
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.start_end_tokens = start_end_tokens
        self.PAD_TOKEN = '<PAD>'

    def build_dictionary(self, data):
        self.vocab_words, self.word2idx, self.idx2word, self.idx2count = self._build_dictionary(data)
        self.vocabulary_size = len(self.vocab_words)

    def indexer(self, word):
        """获取token对应的id"""
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']  # token不存在，返回UNK

    def _build_dictionary(self, data):
        """
        构建词典
        :param data: 数据
        :return: token与id的映射
        """
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2
        if self.start_end_tokens:
            vocab_words += ['<SOS>', '<EOS>']
            vocab_size += 2
        counter = Counter(
            [word for sentence in data for word in sentence.split(' ')])
        # 获取vocab_size的词
        if self.max_vocab_size:
            counter = {
                word: freq
                for word, freq in counter.most_common(self.max_vocab_size -
                                                      vocab_size)
            }
        # 获取大于最小词频的词
        if self.min_count:
            counter = {
                word: freq
                for word, freq in counter.items() if freq >= self.min_count
            }
        vocab_words += list(sorted(counter.keys()))
        # 统计词频
        idx2count = [counter.get(word, 0) for word in vocab_words]
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words
        return vocab_words, word2idx, idx2word, idx2count


if __name__ == '__main__':
    dictionary = Dictionary()
    data = build_dict_dataset()
    word = True  # word粒度或者char粒度
    if word:
        data = data['raw_words'].values.tolist()
    else:
        data = data['raw_words'].apply(lambda x: " ".join("".join(x.split())))
    dictionary.build_dictionary(data)
    joblib.dump(dictionary, config.dict_path)
    print('vocab_words:{}'.format(dictionary.vocab_words[:5]))
    print('word2idx:{}'.format(dictionary.word2idx))
    print('idx2word:{}'.format(dictionary.idx2word[:5]))
    print('idx2count:{}'.format(dictionary.idx2count[:5]))
