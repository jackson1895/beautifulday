import torch.nn as nn
import torch
import numpy as np
import math
from models.Self_Attn import buildSelf_Attn

class Fir_enc(nn.Module):
    def __init__(self, input_size,hidden_size,n_layers,dropout,bidirectional,duration,config):
        super(Fir_enc, self).__init__()
        self.v2h=nn.Linear(input_size,hidden_size)
        self.duration =duration
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.attn = buildSelf_Attn(config)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.v2h.weight)

    def forward(self, inputs,input_lengths):
        assert inputs.dim() == 3
        inputs=self.v2h(inputs)
        vid_duration = inputs.shape[1]
        ret = []
        res_o=[]
        for i in range(0, (vid_duration - self.duration), int(self.duration / 2)):
            tmp_li = []
            for id in range(self.duration):
                tmp_li.append(i + id)
            ret.append(torch.index_select(inputs, 1, torch.LongTensor(tmp_li).cuda()))
        for i in range(len(ret)):
            hidden,_ = self.lstm(ret[i])
            context = self.attn(hidden)
            res_o.append(context)
             # res_o.append(self.lstm(ret[i])[1][0][2:,:,:])
        inputs = torch.stack(res_o,1)
        '''
        没有self_attn的代码
        # inputs = torch.stack(res_o, 2)
        # inputs = torch.mean(inputs,0)
        
        '''

        # print(input_lengths.sub(self.duration))
        # print(torch.ceil(torch.div(input_lengths.sub(self.duration).float(),4.0)).long())
        # print(torch.ceil(input_lengths.sub(self.duration)/torch.tensor([2]).cuda()))
        #length = math.ceil(float(video_duration-duration)/duration/2))
        input_lengths = torch.ceil(torch.div(input_lengths.sub(self.duration).float(),float(self.duration/2))).long()
        return inputs, input_lengths

def buildFir_enc(config):
    if config.type == 'lstm':
        return Fir_enc(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            n_layers=config.n_layers,
            dropout=config.dropout_p,
            bidirectional=config.bidirectional,
            duration=config.duration,
            config = config
        )
    else:
        raise NotImplementedError
