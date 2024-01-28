import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNK = 0                         # 未登录词标识符的索引
PAD = 1                         # padding占位符的索引

BATCH_SIZE = 8                 # 批次大小
EPOCHS = 150                    # 训练轮数
LAYERS = 3                      # transformer中encoder和decoder层数
H_NUM = 4                       # multi head attention hidden个数
D_MODEL = 256                   # embedding 维度
D_FF = 512                      # feed forward第一个全连接层维数
DROPOUT = 0.1                   # dropout比例
MAX_LENGTH = 100                # 语句最大长度，过大测试时速度会很慢
WAIT = 4                        # 早停，等待的最大epoch数

# 数据集 'news-commentary-v15' or 'back-translated-news'
# DATA_SET = 'news-commentary-v15'
DATA_SET = 'back-translated-news'

# 'news-commentary-v15'数据集存储的位置
DATA_FILE = 'data/news-commentary-v15.en-zh.tsv'

# 'back-translated-news'数据集存储的位置
DATA_FILE_EN = '/kaggle/input/translations/news.en'
DATA_FILE_CN = '/kaggle/input/translations/news.translatedto.zh'


SAVE_FILE = 'model.pt'     # 模型保存路径
MODEL_FILE = '/kaggle/input/test-model/model.pt'    # 从训练好的模型迁移，‘train.py’

RES_FILE = "res.txt"
BLEU_REFERENCES = "/kaggle/working/exp5/data/bleu/references.txt" # BLEU评价参考译文
BLEU_CANDIDATE = "/kaggle/working/exp5/data/bleu/candidate.txt"  # 模型翻译译文