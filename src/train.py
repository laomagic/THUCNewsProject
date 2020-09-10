import os
from src.dictionary import Dictionary
from src.utils import config
from src.utils.tools import create_logger, build_dict_dataset
from importlib import import_module
import argparse
import numpy as np
import torch
import joblib
from src.dataset import NewsDataset, collate_fn
from torch.utils.data import DataLoader
from src.train_eval import train, init_network


parse = argparse.ArgumentParser(description='文本分类')
parse.add_argument('--model', type=str, default='TextCNN', help='选择模型: CNN, RNN, RCNN, RNN_Att, DPCNN, Transformer')
parse.add_argument('--word', default=True, type=bool, help='词或者字符')
parse.add_argument('--dictionary', default=config.dict_path, type=str, help='字典的路径')
args = parse.parse_args()


if __name__ == '__main__':
    model_name = args.model
    x = import_module('models.' + model_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    logger = create_logger(config.root_path + '/logs/train.log')

    logger.info('Building dictionary')
    # 构建字典
    if os.path.exists(args.dictionary):
        dictionary = joblib.load(args.dictionary)
    else:
        logger.info("Loading data...")
        data = build_dict_dataset()  # 构建字典数据集
        # 词粒度或者字符粒度
        if args.word:
            data = data['raw_words'].values.tolist()
        else:
            data = data['raw_words'].apply(lambda x: " ".join("".join(x.split())))
        dictionary = Dictionary()
        dictionary.build_dictionary(data)
        del data
        joblib.dump(dictionary, config.dict_path)

    logger.info('Loading dataset')
    # 数据集的定义
    train_dataset = NewsDataset(config.train_path, dictionary=dictionary, word=args.word)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)
    dev_dataset = NewsDataset(config.valid_path, dictionary=dictionary, word=args.word)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=True)
    test_dataset = NewsDataset(config.test_path, dictionary=dictionary, word=args.word)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_fn)
    logger.info('load network')
    model = x.Model(config).to(config.device)
    logger.info('init network')
    # 初始化参数
    init_network(model)
    logger.info('training model')
    train(config, model, train_dataloader, dev_dataloader, test_dataloader, model_name)
