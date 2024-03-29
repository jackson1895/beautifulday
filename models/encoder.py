import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)
        self._init_hidden()


    def _init_hidden(self):
        nn.init.xavier_normal_(self.output_proj.weight)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits, hidden


def build_encoder(config):
    if config.type == 'lstm':
        return BaseEncoder(
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            n_layers=config.n_layers,
            dropout=config.dropout_p,
            bidirectional=config.bidirectional
        )
    else:
        raise NotImplementedError
