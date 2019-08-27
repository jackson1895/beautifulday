import torch.nn as nn
from torch.nn.utils import rnn
import torch
class S2VTAttModel(nn.Module):
    def __init__(self, encoder,Two_Lstm, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.Two_Lstm=Two_Lstm

    def forward(self, vid_feats, target_variable=None,
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
        indices1 = torch.index_select(vid_feats,1,torch.LongTensor([0,1,2,3,4,5,6,7]).cuda())
        indices2 = torch.index_select(vid_feats,1,torch.LongTensor([8,9,10,11,12,13,14,15]).cuda())
        indices3 = torch.index_select(vid_feats,1,torch.LongTensor([16,17,18,19,20,21,22,23]).cuda())
        indices4 = torch.index_select(vid_feats,1,torch.LongTensor([24,25,26,27,28,29,30,31]).cuda())
        indices5 = torch.index_select(vid_feats,1,torch.LongTensor([32,33,34,35,36,37,38,39]).cuda())
        indices6 = torch.index_select(vid_feats,1,torch.LongTensor([40,41,42,43,44,45,46,47]).cuda())
        indices7 = torch.index_select(vid_feats,1,torch.LongTensor([48,49,50,51,52,53,54,55]).cuda())
        indices8 = torch.index_select(vid_feats,1,torch.LongTensor([56,57,58,59,60,61,62,63]).cuda())
        indices9 = torch.index_select(vid_feats,1,torch.LongTensor([64,65,66,67,68,69,70,71]).cuda())
        indices10 = torch.index_select(vid_feats,1,torch.LongTensor([72,73,74,75,76,77,78,79]).cuda())
        M2=torch.zeros(vid_feats.shape[0],10)
        M2_F = torch.zeros(vid_feats.shape[0],10)
        fil,_ = self.encoder(torch.zeros(1,8,512).cuda())
        fil = fil.view(fil.shape[0],-1)
        # fil_mat = fil.repeat(vid_feats.shape[0],1,1)
        o0,h0=self.encoder(indices1)
        o1, h1 = self.encoder(indices2)
        o2, h2 = self.encoder(indices3)
        o3, h3 = self.encoder(indices4)
        o4, h4 = self.encoder(indices5)
        o5, h5 = self.encoder(indices6)
        o6, h6 = self.encoder(indices7)
        o7, h7 = self.encoder(indices8)
        o8, h8 = self.encoder(indices9)
        o9, h9 = self.encoder(indices10)
        clloc_o=[o0,o1,o2,o3,o4,o5,o6,o7,o8,o9]
        clloc_h =[h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0],h7[0],h8[0],h9[0]]
        for o in range(len(clloc_o)):
            o_i=clloc_o[o].contiguous().view(clloc_o[o].shape[0],-1)
            for r_i in range(o_i.shape[0]):
                res = torch.sub(o_i[r_i],fil).squeeze(0)
                res_1=(res==0).nonzero()
                if len(res_1)==0:
                    M2[r_i, o] = 1
                elif res_1[0][0]==0:
                    M2[r_i,o]=0
                else:
                    M2[r_i,o]=1
                #set M2[r_i,o_i] = 1 if o[r_i]==fil else 0
        rms=torch.ones(1,10).cuda().mm(M2.transpose(0,1).cuda())

        #get M2 one-hot matrix
        for i in range(rms.shape[1]):
            j = int(rms[0][i].item()-1)
            M2_F[i][j]=1
        l2_input = torch.stack(clloc_h,2)
        l2_input = torch.mean(l2_input,0)
        # l2_input=l2_input.view(l2_input.shape[1],l2_input.shape[2],-1)

        encoder_outputs,encoder_hidden = self.Two_Lstm(l2_input)
        # M2_F
        M2_F=torch.unsqueeze(M2_F,1).cuda() #batchsize*1*10
        encoder_hidden0 = torch.bmm(M2_F,encoder_outputs).squeeze(1) #batch_size * 1024
        encoder_hidden=(encoder_hidden0,encoder_hidden[1])
        # encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        # encoder_outputs = rnn.pad_packed_sequence(encoder_outputs,batch_first=True)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt)
        return seq_prob, seq_preds
