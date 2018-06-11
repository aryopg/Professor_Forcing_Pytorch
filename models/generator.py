import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from params import *

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_length, dropout_p, output_lang):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.output_lang = output_lang

        self.enc_embedding = nn.Embedding(input_size, hidden_size)
        self.enc_gru = nn.GRU(hidden_size, hidden_size)

        self.dec_embedding = nn.Embedding(output_size, hidden_size)
        self.dec_attn = nn.Linear(hidden_size * 2, max_length)
        self.dec_attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dec_dropout = nn.Dropout(dropout_p)
        self.dec_gru = nn.GRU(hidden_size, hidden_size)
        self.dec_out = nn.Linear(hidden_size, output_size)

    def encoder(self, input, hidden):
        embedded = self.enc_embedding(input).view(1, input.shape[0], -1)
        output = embedded
        output, hidden = self.enc_gru(output, hidden)
        return output, hidden

    def decoder(self, input, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1,0,2)
        embedded = self.dec_embedding(input).view(1, input.shape[0], -1)
        embedded = self.dec_dropout(embedded)

        attn_weights = F.softmax(
            self.dec_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        attn_applied = attn_applied.permute(1,0,2)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.dec_attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.dec_gru(output, hidden)

        output = F.log_softmax(self.dec_out(output[0]), dim=1)
        return output, hidden, attn_weights

    def forward(self, x, y, encoder_hidden, input_length, target_length, criterion, train, teacher_forced):
        batch_size = x.shape[1]
        nllloss = 0
        encoder_outputs = torch.zeros(self.max_length, batch_size, self.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                x[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device)

        decoder_hidden = encoder_hidden

        decoded_outputs = torch.zeros(target_length, batch_size, 1)
        decoded_words = np.empty((batch_size, self.max_length), dtype=np.object_)
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        for i in range(decoded_words.shape[0]):
            for j in range(decoded_words.shape[1]):
                decoded_words[i,j] = '<EOS>'

        if train:
            assert y.shape[0] > 0  and target_length != 0

            for di in range(target_length):
                if teacher_forced:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    nllloss += criterion(decoder_output, y[di].squeeze())
                    topv, topi = decoder_output.data.topk(1)
                    decoder_input = y[di]

                else:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()  # detach from history as input
                    nllloss += criterion(decoder_output, y[di].squeeze())

                for idx, i in enumerate(topi):
                    if i.item() == EOS_token:
                        decoded_words[idx,di] = '<EOS>'
                    else:
                        decoded_words[idx,di] = self.output_lang.index2word[i.item()]

                decoded_outputs[di,:] = topi.detach()

            return nllloss, decoded_outputs, decoded_words
        else:
            decoded_words = []
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.detach()

            return decoded_words, decoder_attentions[:di + 1]

    def initEncoderHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
