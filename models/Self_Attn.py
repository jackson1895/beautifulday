import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """
    def __init__(self, config):
        super(Attention, self).__init__()
        self.input_dim=config.self_attention.input_size
        self.output_dim=config.self_attention.output_size
        # self.WQ_Linear = nn.Sequential(
        #     nn.Linear(self.input_dim, self.output_dim, bias=True),
        #     nn.LayerNorm(self.output_dim),
        #     nn.Tanh()
        # )
        # self.WK_Linear = nn.Sequential(
        #     nn.Linear(self.input_dim, self.output_dim, bias=True),
        #     nn.LayerNorm(self.output_dim),
        #     nn.Tanh()
        # )
        # self.WV_Linear = nn.Sequential(
        #     nn.Linear(self.input_dim, self.output_dim, bias=True),
        #     nn.LayerNorm(self.output_dim),
        #     nn.Tanh()
        # )
        self.WQ_Linear = nn.Linear(self.input_dim, self.output_dim)
        self.WK_Linear = nn.Linear(self.input_dim, self.output_dim)
        self.WV_Linear = nn.Linear(self.input_dim, self.output_dim)
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.WQ_Linear.weight)
        nn.init.xavier_normal_(self.WK_Linear.weight)
        nn.init.xavier_normal_(self.WV_Linear.weight)

    def forward(self, hidden_state):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        Q = torch.tanh(self.WQ_Linear(hidden_state))
        K = torch.tanh(self.WK_Linear(hidden_state))
        V = torch.tanh(self.WV_Linear(hidden_state))
        # Q=self.WQ_Linear(hidden_state)
        # K=self.WK_Linear(hidden_state)
        # V=self.WV_Linear(hidden_state)
        e = torch.bmm(Q,K.permute(0,2,1))
        e = torch.div(e,np.sqrt(self.output_dim))
        # e= e/ torch.sqrt(torch.tensor[self.output_dim])
        alpha = F.softmax(e,dim=-1)
        context = torch.bmm(alpha,V)
        return torch.mean(context,dim=1)


        # batch_size, seq_len, _ = encoder_outputs.size()
        # hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        # inputs = torch.cat((encoder_outputs, hidden_state),
        #                    2).view(-1, self.dim * 2)
        # o = self.linear2(F.tanh(self.linear1(inputs)))
        # e = o.view(batch_size, seq_len)
        # alpha = F.softmax(e, dim=1)
        #
        # context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        # return context

def buildSelf_Attn(config):
    return Attention(config)