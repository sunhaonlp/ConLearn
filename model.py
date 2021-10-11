import math
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        edge_in_hidden = self.linear_edge_in(hidden)
        edge_out_hidden = self.linear_edge_out(hidden)
        input_in = torch.matmul(A[ :, :A.shape[0]], edge_in_hidden) + self.b_iah
        input_out = torch.matmul(A[ :, A.shape[0]: 2 * A.shape[0]], edge_out_hidden) + self.b_oah
        inputs = torch.cat([input_in, input_out],1)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.linear_his_one = nn.Linear(hidden_size, hidden_size*n_heads, bias=True)
        self.linear_his_two = nn.Linear(hidden_size, hidden_size*n_heads, bias=True)
        self.compress_layer = nn.Linear(hidden_size*n_heads, hidden_size)

    def forward(self, seq_hidden_current):
        his_q1 = self.linear_his_one(seq_hidden_current)
        his_q2 = self.linear_his_two(seq_hidden_current)
        his_q1 = torch.cat(his_q1.unsqueeze(0).split(self.hidden_size, dim=-1), dim=0)
        his_q2 = torch.cat(his_q2.unsqueeze(0).split(self.hidden_size, dim=-1), dim=0)
        his_q2_reshaped = his_q2.transpose(1,2)
        att_weights = torch.softmax(torch.bmm(his_q1, his_q2_reshaped), -1)
        seq_hidden_history_attn = (att_weights.unsqueeze(-1) * his_q2.unsqueeze(1)).sum(-2)
        seq_hidden_history_attn = torch.cat(seq_hidden_history_attn.split(1,0), -1).squeeze(0)
        seq_hidden_history = self.compress_layer(seq_hidden_history_attn) + seq_hidden_current
        return seq_hidden_history


class Main_model(Module):
    def __init__(self, opt, n_node):
        super(Main_model, self).__init__()
        self.hidden_size = opt.hidden_size
        self.bert_size = opt.bert_size
        self.glove_size = opt.glove_size

        self.n_node = n_node
        if opt.usebert:
            self.embedding = nn.Embedding(n_node, self.bert_size)
        else:
            self.embedding = nn.Embedding(n_node, self.glove_size)

        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.self_attention = SelfAttention(self.hidden_size, opt.n_head)
        self.fc_1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True))

        if opt.usebert:
            self.linear_one = nn.Linear(self.bert_size, self.hidden_size, bias=True)
        else:
            self.linear_one = nn.Linear(self.glove_size, self.hidden_size, bias=True)

        self.linear_two = nn.Linear(self.hidden_size * 4, 1, bias=True)
        self.dropout = nn.Dropout(p=opt.dropout_p)

        self.init_weight()

        if opt.usebert:
            self.embedding.load_state_dict(
                {'weight': torch.load('fine_tuning/' + opt.dataset + '_model/' + opt.dataset + '_bert_embeddings.pt', map_location={'cuda:2':'cuda:'+ str(opt.device)})})
            if not opt.updatebert:
                self.embedding.weight.requires_grad = False

        if opt.useglove:
            self.embedding.load_state_dict({'weight': torch.load('fine_tuning/' + opt.dataset + '_model/' + opt.dataset + '_glove_embeddings.pt')})
            if not opt.updateglove:
                self.embedding.weight.requires_grad = False

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, input_A, train_instance_1, train_instance_2):
        ## GNN
        inputs_embedding = self.linear_one(self.embedding(input))
        updated_embedding = self.gnn(input_A, inputs_embedding)

        ## Self Attention
        context_aware_embedding = self.self_attention(updated_embedding)
        context_aware_embedding = self.fc_1(context_aware_embedding) + context_aware_embedding

        ## Siamense Network
        tmp_1 = context_aware_embedding[train_instance_1]
        tmp_2 = context_aware_embedding[train_instance_2]
        tmp_1_drop = self.dropout(tmp_1)
        tmp_2_drop = self.dropout(tmp_2)
        q1 = self.fc_2(tmp_1_drop)
        q2 = self.fc_2(tmp_2_drop)
        q1 = self.dropout(q1)
        q2 = self.dropout(q2)
        aggregate_embedding = torch.cat((q1, q2, q1-q2, torch.mul(q1,q2)), 1)
        final = self.linear_two(aggregate_embedding)
        pre = torch.sigmoid(final)

        return pre