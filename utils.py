import numpy as np

from settings import *

def subsequent_mask(size):
    # 生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它最后两维形成一个方阵
    attn_shape = (1, size, size)
    # 然后使用np.ones方法向这个形状中添加1元素，形成上三角阵
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(mask) == 0

def seq_padding(X, padding=0):
    """
    按批次(batch)对数据填充、长度对齐
    """
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    # (注意这里默认padding id是0，相当于是拿<UNK>来做了padding)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])
    
def bleu_candidate(sentence):
    "保存预测的翻译结果到文件中"
    with open(BLEU_CANDIDATE,'a+',encoding='utf-8') as f:
        f.write(sentence + '\n')
        
def bleu_references(data, save_filename=BLEU_REFERENCES):
    """
    保存参考译文到文件中
    """
    writer = open(save_filename,'a+',encoding='utf-8')
    for i in range(len(data.test_cn)):
        text_cn = data.test_cn[i]
        sentence_tap = " ".join(data.cn_index_dict[text_cn[w]] for w in range(1, len(text_cn)-1))
        writer.write(sentence_tap+'\n')
    writer.close()
    
if __name__ == '__main__':
    from data_pre import PrepareData
    data = PrepareData(DATA_FILE)
    bleu_references(data)