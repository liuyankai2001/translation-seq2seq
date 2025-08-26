import torch
from torch import nn
import config

class TrainslationEncoder(nn.Module):

    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.ENCODER_HIDDEN_SIZE,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=config.ENCODER_LAYERS)
    def forward(self,x): # x.shape [batch_size, seq_len]
        embed = self.embedding(x) # [batch_size, seq_len, embeding_dim]
        output,hidden = self.gru(embed) # hidden.shape [layer*direction, batch_size, hidden_size]
        lasthidden_forward = hidden[-2]
        lasthidden_backward = hidden[-1]
        lasthidden = torch.cat((lasthidden_forward,lasthidden_backward),dim=-1)
        return lasthidden

class TrainslationDecoder(nn.Module):

    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.DECODER_HIDDEN_SIZE,
                          batch_first=True)
        self.linear = nn.Linear(in_features=config.DECODER_HIDDEN_SIZE,out_features=vocab_size)


    def forward(self,x,hidden_0): # [batch_size,1]
        embed = self.embedding(x) # [batch_size, 1, embedding_dim]
        output,hidden_n = self.gru(embed,hidden_0) # [batch_size, 1, encoder_hidden_size], [1, batch_size, encoder_hidden_size]
        output = self.linear(output) # [batch_size, 1, vocab_size]
        return output,hidden_n

