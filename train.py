import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_pre import PrepareData
from model import make_model,  SimpleLossCompute, LabelSmoothing, NoamOpt
from settings import *

# 数据预处理
data = PrepareData(DATA_FILE)
src_vocab = len(data.en_word_dict)
tgt_vocab = len(data.cn_word_dict)
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)

# 模型的初始化
model = make_model(src_vocab, tgt_vocab, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)

seed = 10 
random.seed(seed) 
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(seed)

def plot_loss(train_loss_list, dev_loss_list):
    plt.figure()
    plt.plot(train_loss_list, c="red", label="train_loss")
    plt.plot(dev_loss_list, c="blue", label="dev_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss of Train and Validation in each Epoch")
    plt.savefig("loss.png")

def run_epoch(data, model, loss_compute, epoch):
    # start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # 里面包含了 计算loss -> backward -> optimizer.step() -> 梯度清零
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens  # 实际的词数

        # if i % 50 == 0:
        #     elapsed = time.time() - start
        #     print("Loss: %f / Tokens per Sec: %fs" % (loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
        #     start = time.time()
        #     tokens = 0
    return total_loss / total_tokens

def train(data, model, criterion, optimizer):
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5
    delay = 0
    train_loss_list, dev_loss_list = [], []
    for epoch in range(EPOCHS):
        print(f"#" * 50 + f"Epoch: {epoch + 1}" + "#" * 50)
        
        model.train()
        train_loss = run_epoch(data.train_data, model, 
                               SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        
        model.eval()
        # 在dev集上进行loss评估
        dev_loss = run_epoch(data.dev_data, model, 
                             SimpleLossCompute(model.generator, criterion, None), epoch)
        
        train_loss_list.append(train_loss)
        dev_loss_list.append(dev_loss)
        print(f'train loss: {train_loss}, eval loss: {dev_loss}')
        
        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            print(f'Update best_dev_loss to {best_dev_loss:10.6f}')
            best_dev_loss = dev_loss
            delay = 0
            torch.save(model.state_dict(), SAVE_FILE)
            print('Save model done...')
        else:
            delay = delay + 1
        
        # 早停
        if delay > WAIT:
            break
        
    plot_loss(train_loss_list, dev_loss_list)

if __name__ == '__main__':
    # training part
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.1)  # 损失函数
    optimizer = NoamOpt(D_MODEL, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  # 优化器
    train(data, model, criterion, optimizer)  # 训练函数(含保存)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")