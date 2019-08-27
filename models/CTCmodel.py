import torch.nn as nn
import torch
from utils import sort_batch,reverse_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F




class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden,bidirectional=True,batch_first=True,dropout=0.5,num_layers=2)
        self.embedding = nn.Linear(nHidden*2 , nOut)
        self.input_dropout = nn.Dropout(0.2)


    def forward(self, input):
        # input = self.input_dropout(input)
        # dropout_input=self.input_dropout(input.data)
        # x_dropout = nn.utils.rnn.PackedSequence(dropout_input, input.batch_sizes)
        recurrent, _ = self.rnn(input)
        # if isinstance(recurrent,PackedSequence):

        padded_res, _ = nn.utils.rnn.pad_packed_sequence(recurrent, batch_first=True)
        # all_output ,all_length = nn.utils.rnn.pad_packed_sequence(recurrent,batch_first=True)
        # row_indices = torch.arange(0, all_output.size(0)).long()
        # col_indices = all_length - 1
        # last_output = all_output[row_indices, col_indices, :]
        # last_output = all_output[row_indices, col_indices, :].unsqueeze(1)
        # output1 = self.classifier(last_output).squeeze(1)
        # desorted_res = padded_res[desorted_indices]
        b,T, h = padded_res.size()
        t_rec = padded_res.contiguous().view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.contiguous().view(b, T, -1)

        return output


class CTCmodel(nn.Module):

    def __init__(self, nclass, nh):
        super(CTCmodel, self).__init__()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nclass))
            # BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):

        # rnn features
        output = self.rnn(input)
        return output


# class CTCmodel(nn.Module):
#     def __init__(self, dim_vid, dim_hidden, vocab_size,input_dropout_p=0.2, rnn_dropout_p=0.5,
#                  n_layers=2, bidirectional=False, rnn_cell='lstm'):
#         """
#
#         Args:
#             hidden_dim (int): dim of hidden state of rnn
#             input_dropout_p (int): dropout probability for the input sequence
#             dropout_p (float): dropout probability for the output sequence
#             n_layers (int): number of rnn layers
#             rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
#         """
#         super(CTCmodel, self).__init__()
#         self.dim_vid = dim_vid
#         self.dim_hidden = dim_hidden
#         self.input_dropout_p = input_dropout_p
#         self.rnn_dropout_p = rnn_dropout_p
#         self.n_layers = n_layers
#         self.bidirectional = bidirectional
#         self.rnn_cell = rnn_cell
#         self.dim_output=vocab_size
#
#         self.vid2hid = nn.Linear(dim_vid, dim_hidden)
#         #nn.Dropout一般为了防止模型过拟合，用在全连接层中
#         self.input_dropout = nn.Dropout(input_dropout_p)
#
#         if rnn_cell.lower() == 'lstm':
#             self.rnn_cell = nn.LSTM
#         elif rnn_cell.lower() == 'gru':
#             self.rnn_cell = nn.GRU
#
#         self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
#                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)
#         self.out = nn.Linear(self.dim_hidden, self.dim_output)
#         self._init_hidden()
#
#     def _init_hidden(self):
#         nn.init.xavier_normal_(self.vid2hid.weight)
#
#     def forward(self, vid_feats):
#         """
#         Applies a multi-layer RNN to an input sequence.
#         Args:
#             input_var (batch, seq_len): tensor containing the features of the input sequence.
#             input_lengths (list of int, optional): A list that contains the lengths of sequences
#               in the mini-batch
#         Returns: output, hidden
#             - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
#             - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
#         """
#         batch_size, seq_len, dim_vid = vid_feats.size()
#         vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
#         vid_feats = self.input_dropout(vid_feats)
#         vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
#         self.rnn.flatten_parameters()
#         output, hidden = self.rnn(vid_feats)
#         tmp= self.out(output.squeeze(1))
#         probs=F.softmax(tmp,dim=2)
#         return probs





# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, nIn, nHidden, nOut):
#
#         super(BidirectionalLSTM, self).__init__()
#         # super(S2VTModel, self).__init__()
#         # if rnn_cell.lower() == 'lstm':
#         self.rnn_cell = nn.LSTM
#         # elif rnn_cell.lower() == 'gru':
#         #     self.rnn_cell = nn.GRU
#         self.rnn = self.rnn_cell(nIn, nHidden,batch_first=True,bidirectional=True, dropout=0.5)
#         # self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers,
#         #                           batch_first=True, dropout=rnn_dropout_p)
#         # self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
#         #                           batch_first=True, dropout=rnn_dropout_p)
#
#         # self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
#         #                           batch_first=True, dropout=rnn_dropout_p)
#
#         # self.dim_vid = dim_vid
#         # self.dim_output = vocab_size
#         # self.dim_hidden = dim_hidden
#         # self.dim_word = dim_word
#         # self.max_length = max_len
#         # self.sos_id = sos_id
#         # self.eos_id = eos_id
#
#         # self.embedding = nn.Embedding(nOut, self.dim_word)
#         # self.embedding = nn.Embedding(self.dim_output, self.dim_word)
#         self.embedding = nn.Linear(nHidden * 2, nOut)
#         # self.out = nn.Linear(nHidden, nOut)
#
#         # self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
#         # self.embedding = nn.Linear(nHidden * 2, nOut)
#
#     def forward(self, input):
#         batch_size, clips, _ = input.shape
#         # padding_words = Variable(vid_feats.data.new(batch_size, n_frames, self.dim_word)).zero_()
#         # padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.dim_vid)).zero_()
#         state = None
#         # state2 = None
#         # self.rnn1.flatten_parameters()
#         # self.rnn2.flatten_parameters()
#         output,_= self.rnn(input, state)
#         # input2 = torch.cat((output1, padding_words), dim=2)
#         # output2, state2 = self.rnn2(input2, state2)
#
#         # seq_probs = []
#         # seq_preds = []
#         # if mode == 'train':
#         for i in range(self.max_length - 1):
#                 # <eos> doesn't input to the network
#             current_words = self.embedding(target_variable[:, i])
#             self.rnn1.flatten_parameters()
#             self.rnn2.flatten_parameters()
#             output1, state1 = self.rnn1(padding_frames, state1)
#                 # input2 = torch.cat(
#                 #     (output1, current_words.unsqueeze(1)), dim=2)
#                 # output2, state2 = self.rnn2(input2, state2)
#             logits = self.out(output2.squeeze(1))
#             logits = F.log_softmax(logits, dim=1)
#             seq_probs.append(logits.unsqueeze(1))
#         seq_probs = torch.cat(seq_probs, 1)
#
#
#         # recurrent, _ = self.rnn(input)
#         T, b, h = input.size()
#         t_rec = input.view(T * b, h)
#
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#
#         return output

