import csv
from nltk import word_tokenize

from settings import DATA_FILE as path
 
def load_data(path=path, train_ratio=0.8, dev_ratio=0.1):
        """
        读取翻译前(英文)和翻译后(中文)的数据文件
        每条数据都进行分词，然后构建成包含起始符(BOS)和终止符(EOS)的单词(中文为字符)列表
        形式如: en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        train_en, dev_en, test_en = [], [], []
        train_cn, dev_cn, test_cn = [], [], []

        with open(path, mode='r', newline='', encoding='utf-8') as file:
            reader = list(csv.reader(file, delimiter='\t'))
            record_count = sum(1 for row in reader)
            for i in range(0, int(record_count*train_ratio)):
                train_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                train_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][1]])) + ["EOS"])
            for i in range(int(record_count*train_ratio), int(record_count*(train_ratio+dev_ratio))):
                dev_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                dev_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][1]])) + ["EOS"])
            for i in range(int(record_count*(train_ratio+dev_ratio)), record_count):
                test_en.append(["BOS"] + word_tokenize(reader[i][0].lower()) + ["EOS"])
                test_cn.append(["BOS"] + word_tokenize(" ".join([w for w in reader[i][1]])) + ["EOS"])
        print(len(train_en))
        print(len(train_cn))
        print(len(dev_en))
        print(len(dev_cn))
        print(len(test_en))
        print(len(test_cn))
        
        return train_en, train_cn, dev_en, dev_cn, test_en, test_cn

load_data(path=path, train_ratio=0.8, dev_ratio=0.1)