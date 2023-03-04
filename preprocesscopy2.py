

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import math
import numpy as np
import re


def save_data(texts, labels, path):
    all_texts = []
    with open(path, 'w', encoding='utf8') as f:
        for i in range(len(texts)):
            text = re.sub('\n','', texts[i])
            text = re.sub('','\'', text)
            text = re.sub('\[.*\]','', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('â', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('ð', '', text)
            text = re.sub('ï¿½?', '', text)
            text = re.sub('', '', text)

            text = re.sub(' ', '', text)
            text = re.sub('ï', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('#[a-zA-Z]{1,} ', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('', '', text)
            text = re.sub('\.\.', '', text)
            text = re.sub('\.\.\.', '', text)

            if len(text) == 0 or text in all_texts:
                continue

            f.write(text + '\t' + str(labels[i]) + '\n')
            all_texts.append(text)
        f.close()


# 取百分位
def statistic_text_len(texts, bar):
    lens = [len(text.split()) for text in texts]
    return math.ceil(np.percentile(lens, bar))


if __name__ == '__main__':
    # 训练集
    train_data = pd.read_csv('data/train_set.csv', encoding='unicode_escape')
    train_texts = train_data.text.tolist()
    train_labels = train_data.label_sexist.tolist()

    print('过滤前数据条数：{}'.format(len(train_labels)))
    # 过滤非正常的标签
    true_label = ['not sexist', 'sexist']
    train_texts, train_labels = zip(*filter(lambda x: x[1] in true_label, zip(train_texts, train_labels)))
    print('过滤后数据条数：{}'.format(len(train_labels)))
    print(Counter(train_labels))

    labels_set = set(train_labels)
    label_dict = {}
    with open('processed_data/class.txt', 'w', encoding='utf8') as f:
        for i, label in enumerate(labels_set):
            label_dict[label] = i
            f.write(label + '\n')
        f.close()
    labels = [label_dict[label] for label in train_labels]
    train_x, val_x, train_y, val_y = train_test_split(train_texts, labels, test_size=0.2, stratify=labels)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)

    print('训练集数据量：{}'.format(len(train_x)))
    print('验证集数据量：{}'.format(len(val_x)))
    save_data(train_x, train_y, 'processed_data/train.txt')
    save_data(val_x, val_y, 'processed_data/val.txt')

    # test_data = pd.read_csv('data/val_set.csv', encoding='unicode_escape')
    # test_labels = pd.read_csv('data/val_set_label.csv', encoding='unicode_escape')
    # test = pd.merge(test_data, test_labels, on='rewire_id', how='inner')
    # test_x = test.text.tolist()
    # test_y = test.label.tolist()
    # test_y = [label_dict[label] for label in test_y]
    save_data(test_x, test_y, 'processed_data/test.txt')
    print('测试集数据量：{}'.format(len(test_y)))

    max_len = statistic_text_len(train_x, 95)
    print(max_len)
    print('处理完成')