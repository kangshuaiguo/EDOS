# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from focal_loss import focal_loss


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
        self.bert_path = '/Users/shaozhuquan/shao/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.5
        self.rnn_hidden = 768
        self.num_layers = 2
        self.inner_size = 256


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.config.output_hidden_states = True  # 需要设置为true才输出
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.rnn_hidden * 2, config.inner_size)
        self.fc2 = nn.Linear(config.inner_size, config.num_classes)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(config.rnn_hidden * 2)
        self.bn2 = nn.BatchNorm1d(config.rnn_hidden)
        self.loss_fn = focal_loss(alpha=0.25, gamma=1, num_classes=config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls, all_hidden_states = self.bert(context, attention_mask=mask)
        out, (last_hidden, _) = self.lstm(all_hidden_states[10])
        out = torch.cat((last_hidden[-1, :, :,], last_hidden[-2,:,:,]), axis=1)
        out = self.relu(out)
        out = self.bn(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # 句子最后时刻的 hidden state
        return out
