import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import math
import numpy as np
import re


if __name__ == '__main__':
    # 训练集
    
    train_data2 = pd.read_csv('data/EXIST2021_training.tsv', sep='\t', encoding='utf8')
    train_texts2 = train_data2.text.tolist()
    train_labels2 = train_data2.task1.tolist()
    print(train_texts2.shape)

   
   