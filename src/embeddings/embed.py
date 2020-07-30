# coding:utf-8
import pandas as pd
from src.utils.tools import create_logger, clean_symbols, query_cut, rm_stop_word
from src.utils import config
from tqdm import tqdm
import gensim
from gensim import models
from src.utils.tools import timethis
logger = create_logger(config.root_path + '/logs/embedding.log')
tqdm.pandas()


class SingletonMetaclass(type):
    '''
    单例模式
    '''

    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        self.load_data()

    def load_data(self):
        '''
        加载数据集并处理数据
        '''
        logger.info('load data')
        self.data = pd.concat([
            pd.read_csv(config.train_path, sep='\t'),
            pd.read_csv(config.test_path, sep='\t'),
            pd.read_csv(config.valid_path, sep='\t')
        ]).dropna()

        # 获取原始文本信息
        self.data["sentence"] = self.data['title'] + self.data['content']
        self.data['clean_sentence'] = self.data['sentence'].progress_apply(clean_symbols)
        # 去除标点符号
        self.data["cut_sentence"] = self.data['clean_sentence'].progress_apply(query_cut)
        # 去除停用词
        self.data['stop_sentence'] = self.data["cut_sentence"].progress_apply(rm_stop_word)

    @timethis
    def trainer(self):
        '''
        训练词向量
        '''
        logger.info('train word2vec')
        # 训练 word2vec
        self.w2v = gensim.models.Word2Vec(min_count=5,
                                          window=3,
                                          size=300,
                                          sample=6e-5,
                                          alpha=0.03,
                                          min_alpha=0.0007,
                                          negative=15,
                                          workers=-1,
                                          sg=1,
                                          iter=15,
                                          max_vocab_size=50000)

        self.w2v.build_vocab(self.data['stop_sentence'])
        self.w2v.train(self.data['stop_sentence'],
                       total_examples=self.w2v.corpus_count,
                       epochs=50,
                       report_delay=1)

        logger.info('train fast')
        # 训练fast的词向量
        self.fast = gensim.models.FastText(min_count=5,
                                           window=3,
                                           size=300,
                                           sample=6e-5,
                                           alpha=0.03,
                                           min_alpha=0.0007,
                                           negative=15,
                                           workers=-1,
                                           iter=15,
                                           sg=1,
                                           max_vocab_size=50000)
        self.fast.build_vocab(self.data['stop_sentence'].values)
        self.fast.train(self.data['stop_sentence'].values,
                        total_examples=self.fast.corpus_count,
                        epochs=50,
                        report_delay=1)

    @timethis
    def saver(self):
        '''
        模型的保存
        '''
        logger.info('save w2v model')
        self.w2v.wv.save_word2vec_format(config.root_path + '/model/w2v.bin', binary=False)
        self.w2v.save(config.root_path + '/model/w2v.model')

        logger.info('save fast model')
        self.fast.wv.save_word2vec_format(config.root_path + '/model/fast.bin', binary=False)
        self.fast.save(config.root_path + '/model/fast.model')

    def load(self):
        '''
        模型的加载
        '''
        logger.info('load w2v model')
        #         self.w2v = models.KeyedVectors.load_word2vec_format(
        #             root_path + '/model/w2v.bin', binary=False)
        self.w2v = models.Word2Vec.load(config.root_path + '/model/w2v.model')

        logger.info('load fast model')
        #         self.fast = models.KeyedVectors.load_word2vec_format(
        #             root_path + '/model/fast.bin', binary=False)
        self.fast = models.FastText.load(config.root_path + '/model/fast.model')


if __name__ == "__main__":
    em = Embedding()
    em.trainer()
    em.saver()
