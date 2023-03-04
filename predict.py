import pandas as pd
import re
from importlib import import_module
import torch
from utils import build_iterator

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def clean_data(text):
    text = re.sub('\n', '', text)
    text = re.sub('', '\'', text)
    text = re.sub('\[.*\]', '', text)
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
    return text

x = import_module('models.' + 'bert_RNN')
config = x.Config('processed_data')


test_data = pd.read_csv('data/test_set.csv', encoding='unicode_escape')
texts = []
for t in test_data.text.tolist():
    texts.append(clean_data(t))

inputs= []
for t in texts:
    token = config.tokenizer.tokenize(t)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    if len(token) < config.pad_size:
        mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
        token_ids += ([0] * (config.pad_size - len(token)))
    else:
        mask = [1] * config.pad_size
        token_ids = token_ids[:config.pad_size]

    inputs.append((token_ids, 0, 0, mask))

test_iter = build_iterator(inputs, config)

model = x.Model(config)
model.load_state_dict(torch.load('saved_dict/bert_85.61.ckpt', map_location=torch.device('cpu')))  #####

out2 = []
for input, _ in test_iter:
    out = model(input)
    predic = torch.max(out.data, 1)[1].numpy()
    cate = {}
    for i, c in enumerate(config.class_list):
        cate[i] = c

    out2.extend([cate[o] for o in predic])

df_out = pd.DataFrame()
df_out['pre_label'] = out2
df_out.to_csv('out001.csv', index=False)