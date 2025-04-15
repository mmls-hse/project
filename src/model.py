import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)

        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context_vector, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention()

    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)

        context_vector, attn_weights = self.attention(encoder_outputs, hidden[-1])

        embedded_trg = self.embedding(trg)
        context_vector = context_vector.unsqueeze(1).expand(-1, trg.size(1), -1)

        decoder_input = torch.cat([embedded_trg, context_vector], dim=2)

        decoder_outputs, _ = self.decoder(decoder_input, (hidden, cell))

        output = self.fc(decoder_outputs)
        return output