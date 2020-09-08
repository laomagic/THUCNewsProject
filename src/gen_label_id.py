import os
import pandas as pd
import json


def write_label_id(data_path, label_path):
    """
    标签映射为id
    :param data_path: 数据的路径
    :param label_path: label的路径
    :return: None
    """
    data = pd.read_csv(data_path, sep='\t').dropna()
    label = data['label'].unique()
    label2id = dict(zip(label, range(len(label))))
    print('label -> id:{}'.format(label2id))
    json.dump(label2id, open(label_path, 'w', encoding='utf-8'))
    print('writing {} successfully'.format(label_path))


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = root_path + '/data/train.csv'
    label_path = root_path + '/data/label2id.json'
    write_label_id(data_path, label_path)
