import torch.nn as nn
from torch.autograd import Variable


class RNNUnit(nn.Module):

    def __init__(self):
        super(RNNUnit, self).__init__()
        self.rnn_type = 'lstm'#opt.rnn_type
        self.rnn_size = 128#opt.rnn_size
        self.num_layers = 1#opt.num_layers
        self.drop_prob_lm = 0#opt.drop_prob_lm
        self.input_size = 2*17*2*2

        self.rnn = getattr(
            nn,
            self.rnn_type.upper())(
            self.input_size,
            self.rnn_size,
            self.num_layers,
            bias=False)#,
            #dropout=self.drop_prob_lm)

    def forward(self, x_t, state):
        output, state = self.rnn(x_t.unsqueeze(0), state)
        return output.squeeze(0), state


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        self.core = RNNUnit()


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if self.rnn_type == 'lstm':
            return (
                Variable(
                    weight.new(
                        self.num_layers,
                        batch_size,
                        self.rnn_size).zero_()),
                Variable(
                    weight.new(
                        self.num_layers,
                        batch_size,
                        self.rnn_size).zero_()))
        else:
            return Variable(
                weight.new(
                    self.num_layers,
                    batch_size,
                    self.rnn_size).zero_())


    def forward(self, frames_pose_dict):
        frames_num = len(frames_pose_dict[0].keys())
        batch_size = len(frames_pose_dict)
        state = self.init_hidden(batch_size)
        output = None

        for frame_idx in range(0, frames_num):
            curr_frame_pose = frames_pose_dict[str(frame_idx)]
            output, state = self.core(curr_frame_pose, state)

        return output
