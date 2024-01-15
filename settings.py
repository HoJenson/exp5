import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNK = 0                         # 未登录词标识符的索引
PAD = 1                         # padding占位符的索引

BATCH_SIZE = 64                 # 批次大小
EPOCHS = 20                     # 训练轮数
LAYERS = 6                      # transformer中encoder和decoder层数
H_NUM = 8                       # multi head attention hidden个数
D_MODEL = 512                   # embedding 维度
D_FF = 1024                     # feed forward第一个全连接层维数
DROPOUT = 0.1                   # dropout比例
MAX_LENGTH = 60                 # 语句最大长度

# SRC_VOCAB = 5493                # 英文的单词数
# TGT_VOCAB = 3194                # 中文的单词数

DATA_FILE = '/kaggle/working/exp5/data/news-commentary-v15.en-zh.tsv'
# TRAIN_FILE = 'data/train.txt'   # 训练集
# DEV_FILE = 'data/dev.txt'       # 验证集
# TEST_FILE = 'data/test.txt'     # 测试文件
SAVE_FILE = '/kaggle/working/exp5/save/model.pt'     # 模型保存路径(注意如当前目录无save文件夹需要自己创建)

# 这里针对的是DEV文件
BLEU_REFERENCES = "data/bleu/references.txt" # BLEU评价参考译文
BLEU_CANDIDATE = "data/bleu/candidate.txt"  # 模型翻译译文