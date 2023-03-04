# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.txt'  # 训练集
        self.dev_path = dataset + '/val.txt'  # 验证集
        self.test_path = dataset + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]  # 类别名单
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 50  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = 'C:\\moinai\\public\\bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
    #     self.bert = BertModel.from_pretrained(config.bert_path)
    #     for param in self.bert.parameters():
    #         param.requires_grad = True
    #     self.fc = nn.Linear(config.hidden_size, config.num_classes)
    #
    # def forward(self, x):
    #     context = x[0]  # 输入的句子
    #     mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
    #     _, pooled,  = self.bert(context, attention_mask=mask, output_all_encoded_layers=True)
    #     out = self.fc(pooled)
    #     return out
        self.config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.config.output_hidden_states = True  # 需要设置为true才输出
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        input_ids = x[0]
        input_mask = x[2]
        last_hidden_states, pool, all_hidden_states = self.bert(input_ids, attention_mask=input_mask)
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
        h = h / len(self.dropouts)
        return h
