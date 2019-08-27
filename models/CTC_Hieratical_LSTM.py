import torch.nn as nn
from torch.nn.utils import rnn
import torch
from .Attention import Attention
import torch.nn.functional as F

class CTC_Hieratical_LSTM(nn.Module):
    def __init__(self, encoder,Two_Lstm,dim_voc,dim_word,dim_hidden,duration,vid_duration):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """

        super(CTC_Hieratical_LSTM, self).__init__()
        self.duration = duration
        self.vid_duration = vid_duration
        self.dim_voc=dim_voc
        self.dim_word=dim_word
        self.dim_hidden=dim_hidden
        self.encoder = encoder
        # self.decoder = decoder
        self.Two_Lstm=Two_Lstm
        self.embedding = nn.Embedding(dim_voc, dim_word)
        self.attention = Attention(1024)
        # self.out = nn.Linear(dim_hidden*2, dim_voc)
        # self.Two_Lstm = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True, dropout=0.5, num_layers=2)


    def forward(self, vid_feats, targets=None,
                mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        # 1. slice 10 * (batch_size, 8, 512)
        # 2. send each slice to LSTM 10 * (batch_size, 512)
        # 3. set another mask M2(batch_size, 10)
        # 4. if a slice is full of Zero, set the corresponding index of M2 zero
        # 5. LSTM2
        # 6. obtain final result bt *
        indx = 1
        res = []
        res_o = []
        res_total=[]
        #TODO 中国手语256 德国手语128
        for i in range(1, (self.vid_duration - self.duration + 1), int(self.duration/2)):
            tmp_li = []
            for id in range(self.duration):
                tmp_li.append(i+id)
            res.append(torch.index_select(vid_feats, 1,torch.LongTensor(tmp_li).cuda()))
        for i in range(len(res)):
            # res_total.append(self.encoder(res[i][1][0]))
            res_o.append(self.encoder(res[i])[1][0])
        l2_input = torch.stack(res_o, 2)
        l2_input = torch.mean(l2_input,0)



        # M2=torch.zeros(vid_feats.shape[0],30)
        # M2_F = torch.zeros(vid_feats.shape[0],30)
        # fil,_ = self.encoder(torch.zeros(1,16,2048).cuda())
        # fil = fil.view(fil.shape[0],-1)
        # # fil_mat = fil.repeat(vid_feats.shape[0],1,1)
        # o0,h0=self.encoder(indices1)
        # o1, h1 = self.encoder(indices2)
        # o2, h2 = self.encoder(indices3)
        # o3, h3 = self.encoder(indices4)
        # o4, h4 = self.encoder(indices5)
        # o5, h5 = self.encoder(indices6)
        # o6, h6 = self.encoder(indices7)
        # o7, h7 = self.encoder(indices8)
        # o8, h8 = self.encoder(indices9)
        # o9, h9 = self.encoder(indices10)
        # clloc_o=[o0,o1,o2,o3,o4,o5,o6,o7,o8,o9]
        # clloc_h =[h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0],h7[0],h8[0],h9[0]]
        # for o in range(len(clloc_o)):
        #     o_i=clloc_o[o].contiguous().view(clloc_o[o].shape[0],-1)
        #     for r_i in range(o_i.shape[0]):
        #         res = torch.sub(o_i[r_i],fil).squeeze(0)
        #         res_1=(res==0).nonzero()
        #         if len(res_1)==0:
        #             M2[r_i, o] = 1
        #         elif res_1[0][0]==0:
        #             M2[r_i,o]=0
        #         else:
        #             M2[r_i,o]=1
        #         #set M2[r_i,o_i] = 1 if o[r_i]==fil else 0
        # rms=torch.ones(1,10).cuda().mm(M2.transpose(0,1).cuda())
        #
        # get M2 one-hot matrix
        # for i in range(rms.shape[1]):
        #     j = int(rms[0][i].item()-1)
        #     M2_F[i][j]=1
        #
        # 输入y 用的
        # l2_input = torch.stack(res_o,2)
        # l2_input = torch.mean(l2_input,0)
        # res_2step=[]
        # for i in range(1, (len(res_o) - 4 + 1), int(2)):
        #    tmp_li = []
        #    for id in range(4):
        #         tmp_li.append(i+id)
        #     res_2step.append(torch.index_select(l2_input, 1,torch.LongTensor(tmp_li).cuda()).mean(1))
        # total_step = len(res_2step)
        # res_2step = torch.stack(res_2step, 1)
        # out_net = []
        # for i in range(total_step):
        #    if i == 0:
        #         sos = torch.zeros((res_2step.shape[0],self.dim_voc),dtype=torch.float).cuda()
        #         sec_lstm_input = torch.cat([res_2step[:,i,:],sos],dim=1).unsqueeze(1)
        #         sec_lstm_output,hiddenstate = self.Two_Lstm(sec_lstm_input)
        #         y_hot = F.log_softmax(sec_lstm_output.squeeze(1), dim=1)
        #         out_net.append(sec_lstm_output)
        #
        #
        #     else:
        #         sec_lstm_input = torch.cat([res_2step[:,i,:],y_hot],dim=1).unsqueeze(1)
        #         sec_lstm_output, hiddenstate = self.Two_Lstm(sec_lstm_input,hiddenstate)
        #         y_hot = F.log_softmax(sec_lstm_output.squeeze(1), dim=1)
        #         out_net.append(sec_lstm_output)
        #
        # out_net=torch.cat(out_net,1)
        # #return out_net
        return self.Two_Lstm(vid_feats)

