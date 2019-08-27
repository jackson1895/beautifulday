import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import build_encoder
from models.decoder import build_decoder
from models.Fir_enc import buildFir_enc
import torch.nn.functional as F

class VideoModel(nn.Module):
    def __init__(self, config):
        super(VideoModel, self).__init__()
        self.config = config
        self.encoder = build_encoder(config.enc)
        self.fir_enc = buildFir_enc(config.fir_enc)
        #if hiratical lstm or not
        self.fir_enc_or_not = config.fir_enc_or_not

    def forward(self, inputs, inputs_length, targets, targets_length):
        if self.fir_enc_or_not:
            t_inputs, t_inputs_length = self.fir_enc(inputs, inputs_length)
        else:
            t_inputs, t_inputs_length = inputs, inputs_length

        enc_state, _ = self.encoder(t_inputs, t_inputs_length)
        enc_state = enc_state.transpose(0, 1).contiguous()
        loss = F.ctc_loss(enc_state, targets.int(), t_inputs_length.int(), targets_length.int())
        return loss

    def recognize(self,inputs,inputs_length):
        if self.fir_enc_or_not:
            t_inputs,t_inputs_length = self.fir_enc(inputs,inputs_length)
        else:
            t_inputs,t_inputs_length = inputs,inputs_length
        batch_size,length = t_inputs.shape[0],t_inputs.shape[1]
        enc_state, _ = self.encoder(t_inputs, t_inputs_length)
        out = F.softmax(enc_state,dim=2)
        pred = torch.argmax(out,dim=2)
        ctced_preds = []
        for i in range(batch_size):
            tmp = []
            t = pred[i,:]
            for j in range(length):
                if t[j] != 0 and (not (j > 0 and t[j - 1] == t[j])):
                    tmp.append(t[j].cpu())
            ctced_preds.append(torch.stack(tmp,0).numpy())

        return ctced_preds
