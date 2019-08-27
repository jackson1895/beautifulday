import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import build_encoder
from models.decoder import build_decoder
from warprnnt_pytorch import RNNTLoss
from models.Fir_enc import buildFir_enc




class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.forward_layer.weight)
        nn.init.xavier_normal_(self.project_layer.weight)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
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
        self.alpha = config.alpha
        # define encoder
        self.encoder = build_encoder(config.enc)
        self.fir_enc = buildFir_enc(config.fir_enc)
        # define decoder
        self.decoder = build_decoder(config.dec)
        self.max_target_length = config.max_target_length
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
            )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

        self.rnnt = RNNTLoss()
        self.crit = nn.CrossEntropyLoss()

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
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)

        crit_target = F.pad(targets,pad=(0,1,0,0),value=0).view(-1)



        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        crit_input = dec_state.view(dec_state.shape[0]*dec_state.shape[1],dec_state.shape[2])

        crit_loss = self.crit(crit_input,crit_target)

        logits = self.joint(enc_state, dec_state)

        loss = self.rnnt(logits, targets.int(), t_inputs_length.int(), targets_length.int())

        concat_loss = self.alpha*loss+(1-self.alpha)*crit_loss

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

        # def decode(enc_state, lengths):
        #     token_list = []
        #
        #     dec_state, hidden = self.decoder(zero_token)
        #
        #     for t in range(lengths):
        #         while True:
        #             logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
        #             out = F.softmax(logits, dim=0).detach()
        #             pred = torch.argmax(out, dim=0)
        #             pred = int(pred.item())
        #
        #             if pred == 0: break
        #             token_list.append(pred)
        #             token = torch.LongTensor([[pred]])
        #
        #             if enc_state.is_cuda:
        #                 token = token.cuda()
        #
        #             dec_state, hidden = self.decoder(token, hidden=hidden)
        #
        #         return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], t_inputs_length[i])
            results.append(decoded_seq)

        return results
