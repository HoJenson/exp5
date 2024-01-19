import torch
from torch.autograd import Variable
from utils import subsequent_mask
from settings import DEVICE, MAX_LENGTH
import numpy as np
from tqdm import tqdm
from utils import bleu_candidate, update_res

from nltk.translate.bleu_score import corpus_bleu

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def evaluate(data, model):
    """
    在data上用训练好的模型进行预测, 打印模型翻译结果
    """
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in tqdm(range(len(data.test_en))):
            # 打印待翻译的英文句子
            text_en = data.test_en[i]
            en_sent = " ".join([data.en_index_dict[text_en[w]] for w in range(1, len(text_en)-1)])
            # 打印对应的中文句子答案
            text_cn = data.test_cn[i]
            cn_sent = " ".join([data.cn_index_dict[text_cn[w]] for w in range(1, len(text_cn)-1)])

            # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.test_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果句子单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            
            # 打印模型翻译输出的中文句子结果
            # 打印前五十条数据测试数据
            if i < 50:
                update_res(en_sent)
                update_res("".join(cn_sent))
                update_res(" ".join(translation))
                # print("\n" + en_sent)
                # print("".join(cn_sent))
                # print("translation: %s" % " ".join(translation))
            
            bleu_candidate(" ".join(translation))

def evaluate_test(data, model):
    evaluate(data, model)

def read_references():
    """
    预料的refetences计算
    :return: [ [['word','word'],['word','word']]   ]
    """
    result = []
    from settings import BLEU_REFERENCES
    f = open(BLEU_REFERENCES,'r',encoding='utf-8')
    sentences = f.readlines()
    for s in sentences:
        references = []
        references.append(s.strip().split(' '))
        result.append(references)
    f.close()
    return result

def read_candidates():
    result = []
    from settings import BLEU_CANDIDATE
    file = open(BLEU_CANDIDATE, 'r', encoding='utf-8')
    sentences = file.readlines()
    for s in sentences:
        result.append(s.strip().split(' '))
    file.close()
    return result

if __name__ == '__main__':
    from settings import SAVE_FILE, DATA_FILE, MODEL_FILE
    from train import model
    from data_pre import PrepareData
    
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
    data = PrepareData(DATA_FILE)
    evaluate_test(data, model)
    
    references = read_references()
    candidates = read_candidates()
    score = corpus_bleu(references, candidates, weights=(1, 0.2, 0, 0))
    print(score)