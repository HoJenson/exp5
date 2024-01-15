import time
import random
import argparse
import numpy as np

from data_pre import PrepareData
from model import make_model,  SimpleLossCompute, LabelSmoothing, NoamOpt
from settings import *

seed = 10 
random.seed(seed) 
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(seed)

def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # 里面包含了 计算loss -> backward -> optimizer.step() -> 梯度清零
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens  # 实际的词数

        if i % 50 == 0:
            elapsed = time.time() - start
            print("Loss: %f / Tokens per Sec: %fs" % (loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5
    for epoch in range(EPOCHS):
        print(f"#" * 50 + f"Epoch: {epoch + 1}" + "#" * 50)
        print('>Train')
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()
        # 在dev集上进行loss评估
        print('>>>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<<<< Evaluate loss: %f' % dev_loss)
        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
            print('> Save model done...')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    
    # 数据预处理
    data = PrepareData(args.path)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    # 模型的初始化
    model = make_model(src_vocab, tgt_vocab, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)

    # training part
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.1)  # 损失函数
    optimizer = NoamOpt(D_MODEL, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  # 优化器
    train(data, model, criterion, optimizer)  # 训练函数(含保存)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")