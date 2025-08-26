import time
from itertools import chain

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tokenizer import ChineseTokenizer,EnglishTokenizer
import config
from model import TrainslationEncoder,TrainslationDecoder
import dataset


def train_one_epoch(dataloader, encoder, decoder, optimizer, loss_function, device):
    encoder.train()
    decoder.train()
    epoch_total_loss = 0
    for inputs,targets in tqdm(dataloader):
        inputs = inputs.to(device)  # [batch_size, seq_len]
        targets = targets.to(device)    # [batch_size, seq_len]
        optimizer.zero_grad()
        # 编码
        context_vector = encoder(inputs) # [batch_size, 2*hidden_size]
        # 解码
        decoder_input = targets[:,0:1]
        decoder_hidden = context_vector.unsqueeze(0)

        decoder_outputs = [] # [seq_len-1, batch_size, 1, vocab_size]
        for t in range(1,targets.shape[1]):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden)
            decoder_input = targets[:,t:t+1]
            decoder_outputs.append(decoder_output)

        # 预测结果
        decoder_outputs = torch.cat(decoder_outputs,dim=1)
        decoder_outputs = decoder_outputs.reshape(-1,decoder_outputs.shape[-1])
        # 期望值
        decoder_targets = targets[:,1:]
        decoder_targets = decoder_targets.reshape(-1)

        # 计算损失
        loss = loss_function(decoder_outputs,decoder_targets)
        loss.backward()
        optimizer.step()
        epoch_total_loss+=loss.item()
    return epoch_total_loss

def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab_txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab_txt')
    encoder = TrainslationEncoder(vocab_size=zh_tokenizer.vocab_size,padding_index=zh_tokenizer.pad_token_id).to(device)
    decoder = TrainslationDecoder(vocab_size=en_tokenizer.vocab_size,padding_index=en_tokenizer.pad_token_id).to(device)
    dataloader = dataset.get_dataloader()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(params=chain(encoder.parameters(),decoder.parameters()),lr=config.LEARNING_RATE)

    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float('inf')
    for epoch in range(1,1+config.EPOCHS):
        print(f"=========== epoch:{epoch} ===========")
        avg_loss = train_one_epoch(dataloader,encoder,decoder,optimizer,loss_function,device)
        print(f"loss:{avg_loss:.4f}")
        writer.add_scalar("Loss",avg_loss,epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(),config.MODELS_DIR / 'encoder.pt')
            torch.save(decoder.state_dict(),config.MODELS_DIR / 'decoder.pt')
            print("模型保存成功")
        else:
            print("模型无需保存！")


if __name__ == '__main__':
    train()