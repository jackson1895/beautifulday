import torch.nn as nn


class two_lstm(nn.Module):
    def __init__(self, dim_hidden,dim_voc, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=0, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(two_lstm, self).__init__()
        self.dim_input = dim_hidden+dim_voc
        self.dim_hidden = dim_hidden
        self.input_dropout_p = nn.Dropout(input_dropout_p)
        self.rnn_dropout_p = rnn_dropout_p
        self.dim_output=dim_voc
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_hidden, self.dim_hidden, n_layers, batch_first=True,
                                bidirectional=True,dropout=self.rnn_dropout_p)
        self.out = nn.Linear(self.dim_hidden*2, self.dim_output)

        # self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, inputs,hidden=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # batch_size, seq_len, dim_vid = inputs.size()
        # inputs = self.vid2hid(inputs.view(-1, dim_vid))
        # # vid_feats = self.input_dropout(vid_feats)
        # inputs = inputs.view(batch_size, seq_len, self.dim_hidden)

        self.rnn.flatten_parameters()
        # inputs=self.input_dropout_p(inputs)
        output, hidden= self.rnn(inputs) #batchsize*10*dim_hidden
        output = self.out(output)
        return output
