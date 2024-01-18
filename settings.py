import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNK = 0                         # 未登录词标识符的索引
PAD = 1                         # padding占位符的索引

BATCH_SIZE = 64                 # 批次大小
EPOCHS = 100                     # 训练轮数
LAYERS = 3                      # transformer中encoder和decoder层数
H_NUM = 4                       # multi head attention hidden个数
D_MODEL = 256                   # embedding 维度
D_FF = 512                      # feed forward第一个全连接层维数
DROPOUT = 0.1                   # dropout比例
MAX_LENGTH = 60                 # 语句最大长度
WAIT = 4                        # 早停，等待的最大epoch数


DATA_FILE = '/kaggle/working/exp5/data/news-commentary-v15.en-zh.tsv'
SAVE_FILE = '/kaggle/working/exp5/model.pt'     # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
MODEL_FILE = '/kaggle/input/model-pt/model.pt'

RES_FILE = "/kaggle/working/exp5/data/res.txt"
BLEU_REFERENCES = "/kaggle/working/exp5/data/bleu/references.txt" # BLEU评价参考译文
BLEU_CANDIDATE = "/kaggle/working/exp5/data/bleu/candidate.txt"  # 模型翻译译文