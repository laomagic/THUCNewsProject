from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import random
from src.utils import config
import os


def count_text(category_path):
    """
    统计总体语料的分布情况
    :param category_path: 语料路径
    :return: 不同种类的语料字典
    """
    if os.path.exists(category_path):
        # 语料的路径
        category_path = category_path + '/*'  # 匹配所有的目录
        total = {}  # 语料总数
        for dir in glob(category_path):
            total[dir] = len(glob(dir + '/*.txt'))  # 每个类别下文件的数量
        print(total)
        print("文件共{}类,总的文件的数量:{}".format(len(total), sum(total.values())))
    else:
        raise FileExistsError('{}文件的路径不存在'.format(category_path))
    return total


def cut_corpus(path):
    """
    切分语料集 训练集 验证集 测试集，比例0.7:0.15:0.15
    :param path: 语料路径list
    :return: 切分后的数据集和标签
    """
    label = re.findall(r"[\u4e00-\u9fa5]+", path)  # 匹配汉字
    files = glob(path + '/*.txt')  # 匹配txt文件的绝对路径
    # 切分数据集
    train, test = train_test_split(files, test_size=0.3, shuffle=True, random_state=2020)
    valid, test = train_test_split(test, test_size=0.5, shuffle=True, random_state=2021)
    print("train:{} test:{} valid:{}".format(len(train), len(test), len(valid)))
    return train, test, valid, label


def read_data(path, label=None, debug=False, frac=1):
    """
    读取文件中的数据title content
    :param path: 每条语料的路径信息list
    :param debug: 采样模式
    :param frac: 采样的比例
    :return:
    """
    titles = []
    contents = []
    for file in tqdm(path):
        with open(file, 'r', encoding='utf-8') as obj:
            data = obj.readlines()
        title = data[0].strip()
        content = [sen.strip() for sen in data[1:]]
        titles.append(title)
        contents.append(''.join(content))

    title_content = defaultdict(list)

    if len(titles) == len(contents):
        title_content['title'] = titles
        title_content['content'] = contents
        title_content['label'] = [label] * len(titles)
    else:
        raise ValueError('数据titles和contents数量不一致')
    df = pd.DataFrame(title_content, columns=['title', 'content', 'label'])
    if debug:
        # 采样
        df = df.sample(frac=frac, random_state=2020).reset_index(drop=True)
        print('采样的样本数量{}'.format(df.shape[0]))
    return df


def writ_to_csv(dictionary, filename='train'):
    """
    将数据写入csv文件
    :param dictionary: 字典格式
    :return:
    """
    df = pd.DataFrame(dictionary, columns=['title', 'content', 'label'])
    df.to_csv('{}.csv'.format(filename), sep='\t', index=False)
    print()
    print('writing succesfully')


def process(path, filename='train', frac=1):
    """
    读取数据文件将数据写入csv文件 title content label
    :param path: 数据文件的路径dict
    :param filename: 保存文件命名
    :return: None
    """
    print('loading {}'.format(filename))
    sample = []
    for label, data in path.items():
        under_sample = read_data(data, label, debug=True, frac=frac)
        sample.append(under_sample)
    df = pd.concat(sample, axis=0)
    print("{}文件的数据量为:{}".format(filename, df.shape[0]))
    # 保存文件的路径
    save_path = config.root_path + '/data/' + filename + '.csv'
    df.to_csv(save_path, sep='\t', index=False)
    print('{} writing succesfully'.format(save_path))


if __name__ == '__main__':
    category_path = config.data_path
    # 语料的路径
    dir_dict = count_text(category_path)
    train_path = defaultdict(list)
    test_path = defaultdict(list)
    valid_path = defaultdict(list)
    for path in dir_dict.keys():
        # 切分数据集
        train, test, valid, label = cut_corpus(path)
        # 保存数据到字典
        train_path[label[0]] = train
        test_path[label[0]] = test
        valid_path[label[0]] = valid

    process(train_path, filename='train_sa', frac=0.6)
    process(test_path, filename='test_sa', frac=0.5)
    process(valid_path, filename='valid_sa', frac=0.5)
