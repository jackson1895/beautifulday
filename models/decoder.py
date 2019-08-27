import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

        if share_weight:
            self.embedding.weight = self.output_proj.weight
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.output_proj.weight)

    def forward(self, inputs, length=None, hidden=None):

        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)



        return outputs, hidden


def build_decoder(config):
    if config.type == 'lstm':
        return BaseDecoder(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            output_size=config.output_size,
            n_layers=config.n_layers,
            dropout=config.dropout,
            share_weight=config.share_weight
        )
    else:
        raise NotImplementedError
