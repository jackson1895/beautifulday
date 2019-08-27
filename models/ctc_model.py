import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import build_encoder
from models.decoder import build_decoder

from models.Fir_enc import buildFir_enc
import torch.nn.functional as F





class ctcmodel(nn.Module):
    def __init__(self, config):
        super(ctcmodel, self).__init__()
        #build cnn
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding=(2,2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,stride=2)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=(2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,stride=2)
        # )
        self.config = config
        # define encoder
        self.encoder = build_encoder(config)
        self.fir_enc = buildFir_enc(config)




        self.crit = F.CTCLoss()
        #if hiratical lstm or not
        self.fir_enc_or_not = config.fir_enc_or_not

    def forward(self, inputs, inputs_length, targets, targets_length):
        # inputs = inputs.unsqueeze(1)
        # conv1_inputs = self.conv1(inputs)
        # conv2_inputs = self.conv2(conv1_inputs)
        # conv2_inputs = conv2_inputs.squeeze(1)
        if self.fir_enc_or_not:
            t_inputs,t_inputs_length = self.fir_enc(inputs,inputs_length)
        else:
            t_inputs,t_inputs_length = inputs,inputs_length

        enc_state, _ = self.encoder(t_inputs, t_inputs_length)


        loss = self.crit(enc_state, targets.int(), t_inputs_length.int(), targets_length.int())

        return loss
    def recognize(self, inputs, inputs_length):

        # inputs = inputs.unsqueeze(1)
        # conv1_inputs = self.conv1(inputs)
        # conv2_inputs = self.conv2(conv1_inputs)
        # conv2_inputs = conv2_inputs.squeeze(1)
        if self.fir_enc_or_not:
            t_inputs, t_inputs_length = self.fir_enc(inputs, inputs_length)
        else:
            t_inputs, t_inputs_length = inputs, inputs_length
        batch_size = t_inputs.size(0)
        enc_states, _ = self.encoder(t_inputs, t_inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []

            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], t_inputs_length[i])
            results.append(decoded_seq)

        return results
