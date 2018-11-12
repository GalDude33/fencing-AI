import torch.nn as nn
import torch.nn.functional as F
from network.PoseEstimationUtils import NUM_LIMBS


class FencingModel(nn.Module):

    def __init__(self):
        super(FencingModel, self).__init__()
        output_size = 3
        #rnn properties
        self.rnn_type = 'lstm'#opt.rnn_type
        self.rnn_size = 128#opt.rnn_size
        self.num_layers = 1#opt.num_layers
        self.drop_prob_lm = 0.5#opt.drop_prob_lm
        self.input_size = 2 * NUM_LIMBS * 2 * 2

        self.rnn = getattr(
            nn,
            self.rnn_type.upper())(
            self.input_size,
            self.rnn_size,
            self.num_layers,
            bias=False,
            dropout=self.drop_prob_lm)
        self.fc = nn.Linear(self.rnn_size, output_size)


    def forward(self, frames_pose_tensor):
        _frames_pose_tensor = frames_pose_tensor.transpose(0, 1) #**input** of shape `(seq_len, batch, input_size)`
        output, state = self.rnn(_frames_pose_tensor)
        logprobs = F.log_softmax(self.fc(output[-1]), dim=1)
        return logprobs
