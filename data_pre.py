import csv
from collections import Counter

import numpy as np
from nltk import word_tokenize
from torch.autograd import Variable

from settings import *
from utils import seq_padding,subsequent_mask


class PrepareData:
    def __init__(self, file_path):
        # 读取数据 并分词
        self.train_en, self.train_cn, self.dev_en, self.dev_cn, self.test_en, self.test_cn = self.load_data(file_path)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # 单词映射为索引
        self.train_en, self.train_cn = self.word2id(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word2id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        self.test_en, self.test_cn = self.word2id(self.test_en, self.test_cn, self.en_word_dict, self.cn_word_dict, sort=False)

        # 划分batch + padding + mask
        self.train_data = self.split_batch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, BATCH_SIZE)
        self.test_data = self.split_batch(self.test_en, self.test_cn, BATCH_SIZE)

    def load_data(self, path, train_ratio=0.8, dev_ratio=0.1):
        """
        读取翻译前(英文)和翻译后(中文)的数据文件
        每条数据都进行分词，然后构建成包含起始符(BOS)和终止符(EOS)的单词(中文为字符)列表
        形式如: en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        train_en, dev_en, test_en = [], [], []
        train_cn, dev_cn, test_cn = [], [], []

        if DATA_SET == 'news-commentary-v15':
            with open(path, mode='r', newline='', encoding='utf-8') as file:
                reader = list(csv.reader(file, delimiter='\t'))
                record_count = sum(1 for _ in reader)
                for i in range(0, int(record_count*train_ratio)):
                    if len(reader[i][0]) != 0:
                        train_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                        train_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][1]])) + ["EOS"])
                for i in range(int(record_count*train_ratio), int(record_count*(train_ratio+dev_ratio))):
                    if len(reader[i][0]) != 0:
                        dev_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                        dev_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][1]])) + ["EOS"])
                for i in range(int(record_count*(train_ratio+dev_ratio)), record_count):
                    if len(reader[i][0]) != 0:
                        test_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                        test_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][1]])) + ["EOS"])
        
        elif DATA_SET == 'back-translated-news':
            record_count = 10000
            with open(DATA_FILE_EN, mode='r', newline='', encoding='utf-8') as file:
                reader = list(csv.reader(file, delimiter='\t'))
                for i in range(0, int(record_count*train_ratio)):
                    if len(reader[i][0]) != 0:
                        train_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                for i in range(int(record_count*train_ratio), int(record_count*(train_ratio+dev_ratio))):
                    if len(reader[i][0]) != 0:
                        dev_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                for i in range(int(record_count*(train_ratio+dev_ratio)), record_count):
                    if len(reader[i][0]) != 0:
                        test_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
            
            with open(DATA_FILE_CN, mode='r', newline='', encoding='utf-8') as file:
                reader = list(csv.reader(file, delimiter='\t'))
                for i in range(0, int(record_count*train_ratio)):
                    if len(reader[i][0]) != 0:
                        train_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][0]])) + ["EOS"])
                for i in range(int(record_count*train_ratio), int(record_count*(train_ratio+dev_ratio))):
                    if len(reader[i][0]) != 0:
                        dev_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][0]])) + ["EOS"])
                for i in range(int(record_count*(train_ratio+dev_ratio)), record_count):
                    if len(reader[i][0]) != 0:
                        test_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][0]])) + ["EOS"])

        return train_en, train_cn, dev_en, dev_cn, test_en, test_cn

    def build_dict(self, sentences, max_words=50000):
        """
        传入load_data构造的分词后的列表数据
        构建词典，即单词-索引映射(key为单词,value为id值)
        """
        # 对数据中所有单词进行计数
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        # 只保留最高频的前max_words数的单词构建词典
        # 添加上UNK和PAD两个单词，对应id已经初始化设置过
        ls = word_count.most_common(max_words)
        # 统计词典的总词数
        # 加2是添加unknow和padding符号
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        # 再构建一个反向的词典，供id转单词使用（id2vec）
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        该方法可以将翻译前(英文)数据和翻译后(中文)数据的单词列表表示的数据均转为id列表表示的数据
        `sort=True`表示以英文语句长度排序，以便按批次填充时，同批次语句填充尽量少
        """
        # 计算英文数据条数
        # length = len(en)
        
        # 将翻译前(英文)数据和翻译后(中文)数据都转换为id表示的形式
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # 构建一个按照句子长度排序的函数
        def len_argsort(seq):
            """
            传入一系列语句数据(分好词的列表形式)
            按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            # 以英文句子长度排序的(句子下标)顺序为基准
            sorted_index = len_argsort(out_en_ids)
            # 对翻译前(英文)数据和翻译后(中文)数据都按此基准进行排序
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        """
        将以单词id列表表示的翻译前(英文)数据和翻译后(中文)数据按照指定的batch_size进行划分
        如果shuffle参数为True, 则会对这些batch数据顺序进行随机打乱
        """
        # 在按数据长度生成的各条数据下标列表[0, 1, ..., len(en)-1]中
        # 每隔指定长度(batch_size)取一个下标作为后续生成batch的起始下标
        idx_list = np.arange(0, len(en), batch_size)
        # 如果shuffle参数为True，则将这些各batch起始下标打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放各个batch批次的句子数据索引下标
        batch_indexs = []
        for idx in idx_list:
            # 注意，起始下标最大的那个batch可能会超出数据大小
            # 因此要限定其终止下标不能超过数据大小
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        # 按各batch批次的句子数据索引下标，构建实际的单词id列表表示的各batch句子数据
        batches = []
        for batch_index in batch_indexs:
            # 按当前batch的各句子下标(数组批量索引)提取对应的单词id列表句子表示数据
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前batch的各个句子都进行padding对齐长度
            # 维度为：batch数量×batch_size×每个batch最大句子长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前batch的英文和中文数据添加到存放所有batch数据的列表中
            # Batch类用于实现注意力掩码
            batches.append(Batch(batch_en, batch_cn))
        return batches


# Object for holding a batch of data with mask during training.
class Batch:
    """
    批次类
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """
    def __init__(self, src, trg=None, pad=0):
        # 将输入与输出的单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        # 对于当前输入的句子非空部分进行判断,变成bool序列
        # 并在seq_len前面增加一维，形成维度为 batch_size×1×seq_len 的矩阵
        # src_mask值为0的位置是False
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
